# Import ---------------------------------------------------------------------------------------------------------------
# data handling
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import pickle
import os
from os import listdir
from os.path import isfile, join
from itertools import chain

# modeling
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.pipeline import Pipeline

# balancing class
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# evaluation
from sklearn.metrics import auc, f1_score, recall_score, accuracy_score, log_loss
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, classification_report
from time import time

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Flask
import flask, json, csv
from flask import request, jsonify
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

sns.set(style='ticks')  # http://seaborn.pydata.org/tutorial/color_palettes.html
color = sns.color_palette()

def get_jsonparsed_data(url):
    """
    Receive the content of ``url``, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)

# Preprocessing --------------------------------------------------------------------------------------------------------
class preprocessing:
    def __init__(self, pwd, num_list, mdr_type):
        self.pwd = pwd
        self.num_list = num_list
        self.mdr_type = mdr_type
        self.threshold = None
        self.max_list = None
        self.min_list = None

    # import and preprocess data
    def input_data(self, threshold=None):
        # read file and split it to x and y
        x = pd.read_csv(self.pwd)
        x.fillna('NA', inplace=True)
        y = x[self.mdr_type]
        x.drop([item for item in x if "MDR_TYPE" in item], axis=1, inplace=True)

        # add the columns which have the word 'useday' to the num_list
        temp = [index for index, value in enumerate(x) if "USEDAY" in value]
        temp = x.columns[temp]
        self.num_list.extend(temp)

        # manage feature types
        temp = [col for col in x.columns if col not in self.num_list]

        x[self.num_list[0]] = x[self.num_list[0]].astype("float64")
        x[self.num_list[1:]] = x[self.num_list[1:]].astype("int64")
        x[temp] = x[temp].astype("object")

        # one-hot-encode categorical columns
        x = pd.get_dummies(x)

        # apply nearZeroVar()
        if threshold is not None:
            x = self.nearZeroVar(x, threshold)
        return x, y

    # delete features whose variance are under threshold
    def nearZeroVar(self, x, threshold):
        selector = VarianceThreshold(threshold=threshold)
        selector.fit_transform(x)
        temp = x.columns[selector.get_support(indices=True)]
        print("There are", str(len(selector.get_support(indices=True))) + " features whose variance is over",
              self.threshold)
        return x.loc[:, temp]

    # normalize numeric columns of train data
    def scale(self, x):
        num_list = self.num_list
        scale_quant = QuantileTransformer()
        x2 = scale_quant.fit_transform(x[num_list])
        pickle.dump(scale_quant, open("/disk2/MDR_ML/scaler/scale_quant.sav", 'wb'))
        scale_minmax = MinMaxScaler()
        x2 = scale_minmax.fit_transform(x2)
        pickle.dump(scale_minmax, open("/disk2/MDR_ML/scaler/scale_minmax.sav", 'wb'))
        x2 = pd.DataFrame(x2, columns=num_list, index=x.index)
        x2 = pd.concat([x2, x.drop(num_list, axis=1)], axis=1)
        return x2

    # normalize numeric columns of test data
    def use_scale(self, x):
        scale_quant = pickle.load(open("/disk2/MDR_ML/scaler/scale_quant.sav", 'rb'))
        x2 = scale_quant.transform(x.loc[:, self.num_list])
        scale_minmax = pickle.load(open("/disk2/MDR_ML/scaler/scale_minmax.sav", 'rb'))
        x2 = scale_minmax.transform(x2)
        x2 = pd.DataFrame(x2, columns=self.num_list, index=x.index)
        x2 = pd.concat([x2, x.drop(self.num_list, axis=1)], axis=1)
        return x2


# Modeling -------------------------------------------------------------------------------------------------------------
class modeling:
    def __init__(self, x_train, x_test, y_train, y_test, model_name, sample_method, random_num, fold_num):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name
        self.sample_method = sample_method
        self.random_num = random_num
        self.fold_num = fold_num


    # get the outputs of model
    def use_model(self, model, x, y):
        y_pred_prob = model.predict_proba(x)[:, 1]
        roc = roc_curve(y, y_pred_prob)
        pr = precision_recall_curve(y, y_pred_prob)
        try:
            f_imp = model.feature_importances_
        except AttributeError:
            f_imp = None
        return y_pred_prob, roc, pr, f_imp


# Main -----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Create API with Flask

    app = flask.Flask(__name__)
    app.config["DEBUG"] = True

    @app.route('/', methods=['GET'])
    def home():
        return(
            '''<h1>3A Machine Learning Prediction Service</h1>
                    <p>Specify MDR Type by</p>
                    <p>/ML?type=MRSA</p>
                    <p>or</p>
                    <p>/ML?type=VRE</p>'''
        )

    @app.route('/ML', methods=['POST'])
    def mainprocedure():
        inputValue = request.form
        print(inputValue)

        if 'type' in inputValue:
            whatuserwants = str(inputValue['type'])

            # whose model do you want to fit?
            mdr = "MDR_TYPE_" + str(inputValue['type'])

            # numeric features except use_days
            num_list = ["AGE_YR", "PRIOR_LOS", "OP_VISIT_1YR", "CURR_LOS"]


            pwd = "./"

            # Get new data and merge
            inputValue=inputValue.to_dict(flat=False)
            jsdf = pd.DataFrame.from_dict(inputValue)
            jsdf = jsdf.drop('type', axis=1)

            # getDummy
            mdrdata = pd.DataFrame([
                {'AGE_YR': 0.0,
                 'ALCOH_DIAG_YN': 'Y',
                 'AMIKA_DRUG_YN': 'Y',
                 'AMINO_PNCILN_DRUG_YN': 'Y',
                 'ANGIO_ORDR_YN': 'Y',
                 'ARTER_CATH_ORDR_YN': 'Y',
                 'CEFTR_DRUG_YN': 'Y',
                 'CLINDA_DRUG_YN': 'Y',
                 'CURR_LOS': 0,
                 'DIABET_DIAG_YN': 'Y',
                 'DIAL_ORDR_YN': 'Y',
                 'DRAIN_TUBE_ORDR_YN': 'Y',
                 'FEED_TUBE_ORDR_YN': 'Y',
                 'PNEU_TBRC_DIAG_YN': 'Y',
                 'FOLEY_CATH_ORDR_YN': 'Y',
                 'GASTRO_DIAG_YN': 'Y',
                 'HEART_DIAG_YN': 'Y',
                 'HYPOALBU_SYMT_YN': 'Y',
                 'HYPRTN_DIAG_YN': 'Y',
                 'ICU_STAY': 'Y',
                 'IMIPE_MEROPE_DRUG_YN': 'Y',
                 'IMMUNO_SPP_DRUG_YN': 'Y',
                 'IV_CATH_ORDR_YN': 'Y',
                 'KIDNEY_DIAG_YN': 'Y',
                 'LINE_DRUG_YN': 'Y',
                 'LTCF_YN': 'Y',
                 'MAL_CANCER_YN': 'Y',
                 'MARR_HEMATO_YN': 'Y',
                 'MTRONIDA_DRUG_YN': 'Y',
                 'NASO_TUBE_ORDR_YN': 'Y',
                 'NEURO_DIAG_YN': 'Y',
                 'OP_VISIT_1YR': 0,
                 'PIPERA_TAZOBAC_DRUG_YN': 'Y',
                 'POLYTRAUMA_DIAG_YN': 'Y',
                 'PRIOR_LOS': 0,
                 'READM_30D_YN': 'Y',
                 'RESPI_DIAG_YN': 'Y',
                 'SEX': 'F',
                 'SHOCK_DIAG_YN': 'Y',
                 'SMOKE_YN': 'Y',
                 'SRGY_YN': 'Y',
                 'SSTI_DIAG_YN': 'Y',
                 'SUCRALF_DRUG_YN': 'Y',
                 'THORA_LUMBER_YN': 'Y',
                 'TIGE_DRUG_YN': 'Y',
                 'TRACH_SRGY_YN': 'Y',
                 'TRACH_TUBE_ORDR_YN': 'Y',
                 'TRNSF_YN': 'Y',
                 'UTI_DIAG_YN': 'Y',
                 'VANCO_DRUG_YN': 'Y',
                 'VASOP_DRUG_YN': 'Y',
                 'VENTIL_TUBE_ORDR_YN': 'Y'},
                {'AGE_YR': 0.0,
                 'ALCOH_DIAG_YN': 'N',
                 'AMIKA_DRUG_YN': 'N',
                 'AMINO_PNCILN_DRUG_YN': 'N',
                 'ANGIO_ORDR_YN': 'N',
                 'ARTER_CATH_ORDR_YN': 'N',
                 'CEFTR_DRUG_YN': 'N',
                 'CLINDA_DRUG_YN': 'N',
                 'CURR_LOS': 0,
                 'DIABET_DIAG_YN': 'N',
                 'DIAL_ORDR_YN': 'N',
                 'DRAIN_TUBE_ORDR_YN': 'N',
                 'FEED_TUBE_ORDR_YN': 'N',
                 'PNEU_TBRC_DIAG_YN': 'N',
                 'FOLEY_CATH_ORDR_YN': 'N',
                 'GASTRO_DIAG_YN': 'N',
                 'HEART_DIAG_YN': 'N',
                 'HYPOALBU_SYMT_YN': 'N',
                 'HYPRTN_DIAG_YN': 'N',
                 'ICU_STAY': 'N',
                 'IMIPE_MEROPE_DRUG_YN': 'N',
                 'IMMUNO_SPP_DRUG_YN': 'N',
                 'IV_CATH_ORDR_YN': 'N',
                 'KIDNEY_DIAG_YN': 'N',
                 'LINE_DRUG_YN': 'N',
                 'LTCF_YN': 'N',
                 'MAL_CANCER_YN': 'N',
                 'MARR_HEMATO_YN': 'N',
                 'MTRONIDA_DRUG_YN': 'N',
                 'NASO_TUBE_ORDR_YN': 'N',
                 'NEURO_DIAG_YN': 'N',
                 'OP_VISIT_1YR': 0,
                 'PIPERA_TAZOBAC_DRUG_YN': 'N',
                 'POLYTRAUMA_DIAG_YN': 'N',
                 'PRIOR_LOS': 0,
                 'READM_30D_YN': 'N',
                 'RESPI_DIAG_YN': 'N',
                 'SEX': 'M',
                 'SHOCK_DIAG_YN': 'N',
                 'SMOKE_YN': 'N',
                 'SRGY_YN': 'N',
                 'SSTI_DIAG_YN': 'N',
                 'SUCRALF_DRUG_YN': 'N',
                 'THORA_LUMBER_YN': 'N',
                 'TIGE_DRUG_YN': 'N',
                 'TRACH_SRGY_YN': 'N',
                 'TRACH_TUBE_ORDR_YN': 'N',
                 'TRNSF_YN': 'N',
                 'UTI_DIAG_YN': 'N',
                 'VANCO_DRUG_YN': 'N',
                 'VASOP_DRUG_YN': 'N',
                 'VENTIL_TUBE_ORDR_YN': 'N'}
            ])
            jsdf = pd.concat([jsdf, mdrdata])
            jsdf.to_csv(pwd + 'newdata.csv', index=False)

            # Get new data
            newD = preprocessing(pwd+'newdata.csv', num_list, mdr)
            x_new, y_new = newD.input_data(threshold=None)
            x_new = newD.use_scale(x_new)
            x_new = x_new.iloc[:1,]

            # Predict probabilities
            mdd = '/disk2/MDR_ML/modelcv/lgb_RUS_'
            prd = []

            for i in range(5) :
                tmpmdd = mdd + str(i) + '.sav'
                temp = pickle.load(open(tmpmdd, 'rb'))
                prd.append(temp.predict_proba(x_new.iloc[:1, :])[:, 1])
            os.remove(pwd+'newdata.csv')
            #return jsonify([np.mean(prd)])
            tmpnum=np.mean(prd)
            return jsonify({'mdrPctg': tmpnum})

        else:
            return jsonify({ 'mdrType':'Error: No MDR type specified.'})
    app.run(host='10.178.113.219', port=5000) # prod
    #app.run(host='169.56.89.85', port=5000)  # dev