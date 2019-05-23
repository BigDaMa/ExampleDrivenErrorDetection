from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import operator

from xgboost import plot_tree

class XGBoostClassifier(object):
    name = 'XGBoost'
    def __init__(self, X_train, X_test, balance=False, feature_names=None):
        self.params = {}
        self.model = {}

        self.name = XGBoostClassifier.name

        self.X_train = xgb.DMatrix(X_train.todense())

        self.balance = balance
        self.feature_names = feature_names

        self.x_alll = None
        self.all_shape = None



    def run_cross_validation(self, train, train_target, folds):
        cv_params = {'min_child_weight': [1, 3, 5],
                     'subsample': [0.7, 0.8, 0.9],
                     'max_depth': [3, 5, 7],
                     'n_estimators': [100, 1000]
                     }

        ind_params = {  # 'min_child_weight': 1, # we could optimize this: 'min_child_weight': [1, 3, 5]
            'learning_rate': 0.1,  # we could optimize this: 'learning_rate': [0.1, 0.01]
            'colsample_bytree': 0.8,
            'silent': 1,
            'seed': 0,
            'objective': 'binary:logistic',
            'n_jobs': 20
        }

        if self.balance:
            ratio = float(np.sum(train_target == False)) / np.sum(train_target == True)
            print "weight ratio: " + str(ratio)
            ind_params['scale_pos_weight'] = ratio

        optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                     cv_params,
                                     scoring='f1', cv=folds, n_jobs=1, verbose=0)

        print train.shape

        optimized_GBM.fit(train, train_target)

        # print "best scores: " + str(optimized_GBM.grid_scores_)

        our_params = ind_params.copy()
        our_params.update(optimized_GBM.best_params_)

        self.params = our_params

    def train_predict_all(self, x, y):
        if self.balance:
            ratio = float(np.sum(y == False)) / np.sum(y == True)
            print "weight ratio: " + str(ratio)
            self.params['scale_pos_weight'] = ratio

        xgdmat = xgb.DMatrix(x.todense(), y)
        self.model = xgb.train(self.params, xgdmat, num_boost_round=3000, verbose_eval=False)

        #feature_importances = self.model.get_score(importance_type='gain')
        #sorted_x = sorted(feature_importances.items(), key=operator.itemgetter(1))
        #print(sorted_x)


        # predict
        probability_prediction = self.model.predict(self.X_train)
        class_prediction = (probability_prediction > 0.5)

        return probability_prediction, class_prediction

    '''
    def explain_prediction(self, x, column_id, feature_names):
        from eli5.explain import explain_prediction
        params = {}
        params['feature_names'] = feature_names
        params['top'] = 5
        expl = explain_prediction(self.model[column_id], x, **params)
        from eli5.formatters import format_as_text
        params_text = {}
        params_text['show_feature_values'] = True
        return format_as_text(expl, **params_text)
    '''


    def predict(self, column_id):
        probability_prediction = self.model[column_id].predict(self.X_test)
        class_prediction = (probability_prediction > 0.5)

        return probability_prediction, class_prediction

    def run_cross_validation_eval(self, train, train_target, folds, column_id):
        scores = cross_val_score(xgb.XGBClassifier(**self.params[column_id]), train, train_target, cv=folds, scoring='f1')
        return scores