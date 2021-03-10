import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os
from os import listdir
from os.path import isfile, join
import datetime

class preprocessing:
    def __init__(self, pwd, files, mdr_list):
        self.pwd = pwd
        self.files = files
        self.mdr_list = mdr_list


    # ai data cleansing
    def ai_cleansing(self):
        # read data
        # CSVFile_STTEST_2018120
        ainputall10 = pd.read_csv(self.pwd + 'CSVFile_MDR_origin_181115.csv', encoding='euc_kr', keep_default_na=False, na_values="")
        ainputall10 = ainputall10.drop_duplicates()

        # delete 00:00:00
        ainputall10.MED_YMD = ainputall10.MED_YMD.apply(lambda x: x[:10])

        # select data
        ainputall10 = ainputall10[ainputall10.MED_YMD >= "2018-06-25"]

        # delete .0
        #ainputall10.PTNT_UUID = ainputall10.PTNT_UUID.apply(lambda x: str(x)[:-2])

        # delete records who isn't younger than 65 years but could be pregnant
        temp = [not row for row in (ainputall10.AGE_YR >= 55) & (ainputall10.PREG_YN == "Y")]
        ainputall10 = ainputall10.loc[temp, :]

        # delete records who isn't older than 3 years but could be pregnant
        temp = [not row for row in (ainputall10.AGE_YR <= 13) & (ainputall10.PREG_YN == "Y")]
        ainputall10 = ainputall10.loc[temp, :]

        # make primary key using 'ID' and 'DATE'
        ai_key = [str(col1) + "_" + col2 for col1, col2 in zip(ainputall10.PTNT_UUID, ainputall10.MED_YMD)]
        ainputall10 = pd.concat([ainputall10, Series(ai_key, name="KEY", index=ainputall10.index)], axis=1)

        # stData
        stData = pd.read_csv(self.pwd + 'st_data_till1212.csv', encoding='euc_kr', keep_default_na=False, na_values="")
        st_key = [str(col1) + "_" + col2 for col1, col2 in zip(stData.PTNT_UUID, stData.MED_YMD)]
        stData = pd.concat([stData, Series(st_key, name="KEY", index=stData.index)], axis=1)

        temp = [row for row in ainputall10.KEY.isin(stData.KEY)]
        ainputall10 = ainputall10.loc[temp, :]

        return ainputall10

    # ai-mdr-match data cleansing
    def multi_label_MDR(self, aist10):
        # read data
        aimdrcheck = pd.read_csv(self.pwd + 'aimdrcheck_20181212.csv', encoding='euc_kr', keep_default_na=False,
                                 na_values="")
        aimdrcheck.MED_YMD = aimdrcheck.MED_YMD.apply(lambda x: x[:10])
        # make MDR_TYPE variable dummy and delete 'VRSA'
        temp = pd.get_dummies(aimdrcheck.MDR_TYPE)
        #temp.drop(["VRSA"], axis=1, inplace=True)
        aimdrcheck = pd.concat([aimdrcheck.iloc[:, :2], temp], axis=1)

        # aggregate mdr state by key
        aimdrcheck = aimdrcheck.groupby(["PTNT_UUID", "MED_YMD"]).agg({key: sum for key in self.mdr_list})
        aimdrcheck.reset_index(level=["PTNT_UUID", "MED_YMD"], inplace=True)

        # make primary key using 'ID' and 'DATE'
        aimdr_key = [str(col1) + "_" + col2 for col1, col2 in zip(aimdrcheck.PTNT_UUID, aimdrcheck.MED_YMD)]
        aimdrcheck = pd.concat([aimdrcheck, Series(aimdr_key, name="KEY", index=aimdrcheck.index)], axis=1)

        # add patients' mdr-state to aist10 data
        temp = [not row for row in aist10.KEY.isin(aimdrcheck.KEY)]
        temp1 = aist10.loc[temp, :]
        temp = {key: np.zeros(temp1.shape[0], dtype='uint8') for key in self.mdr_list}
        temp = pd.DataFrame(temp, index=temp1.index)
        temp1 = pd.concat([temp1, temp], axis=1)

        temp = [row for row in aist10.KEY.isin(aimdrcheck.KEY)]
        temp2 = aist10.loc[temp, :]
        temp2 = pd.merge(temp2, aimdrcheck.iloc[:, 2:], on='KEY')

        aist10mdrchecked = pd.concat([temp1, temp2], axis=0)

        return aist10mdrchecked

    # calculate the duration of hopital stay
    def differ(self, vec):
        l = len(vec)
        null = np.zeros(l)
        if l > 1:
            for i in range(1, l):
                temp = vec[i] - vec[i - 1]
                null[i] = temp.days
            return null
        else:
            return null

    # make inputdata using RFPool
    def make_inputdata(self, aist10mdrchecked, mdr):

        # delete outliers
        # inputdata = aist10mdrchecked.loc[aist10mdrchecked.AGE_YR > 0, :]
        inputdata = aist10mdrchecked.loc[aist10mdrchecked.AGE_YR >= 70, :]

        # rename features on 'MDR_TYPE'
        inputdata.rename(index=str, columns={item: "MDR_TYPE_" + item for item in self.mdr_list}, inplace=True)

        # risk factor pool for each MDR_TYPE
        RFPool = pd.read_csv(self.pwd + 'RFpool_190129.csv', encoding='euc_kr', keep_default_na=False, na_values="")

        ch = ["PTNT_UUID", "KEY"]
        # select features using RFPool data
        if mdr is "MRSA":
            ch.extend(RFPool.loc[RFPool.MRSA.isna(), 'var'])
        else:
            print("Enter correct word.")
            return None
        inputdata = inputdata.loc[:, ch]

        # manage 'DATE' type
        inputdata.MED_YMD = pd.to_datetime(inputdata.MED_YMD, format='%Y/%m/%d')

        # get the duration of hopital stay by 'ID' using differ
        temp = np.argsort(inputdata.MED_YMD)
        inputdata = inputdata.iloc[temp, :]
        inputdata['MDdiff'] = inputdata.groupby('PTNT_UUID')['MED_YMD'].transform(self.differ)
        inputdata.MDdiff = inputdata.MDdiff.astype("int64")

        # leave the records whose 'MDdiff' is over 30 days
        temp1 = inputdata.loc[inputdata.MDdiff > 30, :]

        # select the first visit of each patient who stay in hospital less than 30 days
        temp2 = inputdata.loc[inputdata.MDdiff <= 30, :]

        temp3 = temp2.groupby('PTNT_UUID').apply(lambda x: min(x.MED_YMD))
        temp3.rename("MED_YMD", inplace=True)

        temp = [str(col1) + "_" + str(col2.date()) for col1, col2 in zip(temp3.index, temp3)]
        temp3 = pd.concat([temp3, Series(temp, name="KEY", index=temp3.index)], axis=1)

        temp = temp2.loc[temp2.KEY.isin(temp3.KEY), :]

        # merge two tables and delete unnecessary features for machine learning model
        inputdata = pd.concat([temp1, temp], axis=0)
        inputdata.drop(["KEY", "PTNT_UUID", "MED_YMD", "MDdiff"], axis=1, inplace=True)
        return inputdata


if __name__ == "__main__":
    # present working directory for raw data
    pwd_data = "./rawData/"

    # present working directory for cleaned data
    pwd_mdr = "./MDR_data/"

    # list of files on present working directory for raw data
    cwd = os.getcwd()
    files = [f for f in listdir(cwd + pwd_data) if isfile(join(cwd + pwd_data, f))]

    # mdr-list which you want to deal with
    mdr_list = ["MRSA", "MRAB", "MRPA", "VRE", "CRE"]


    # which of them do you want to make?
    target = ["MRSA"]

    # what day is it today?
    date = "20190201_val"

    # generate objects using 'preprocessing' class
    data = preprocessing(pwd_data, files, mdr_list)
    aist10 = data.ai_cleansing()
    aist10mdrchecked = data.multi_label_MDR(aist10)

    # write cleaned data to csv file
    for mdr in target:
        inputdata = data.make_inputdata(aist10mdrchecked, mdr)
        inputdata.to_csv(cwd + pwd_mdr + date + "_" + mdr + ".csv", index=False)