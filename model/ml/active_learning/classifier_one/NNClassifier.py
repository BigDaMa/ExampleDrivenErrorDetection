from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import keras
import tensorflow as tf
import pickle

class NNClassifier(object):
    name = 'NN'

    def __init__(self, X_train, X_test, use_scale=True, feature_names=None):
        self.params = {}
        self.model = {}
        self.use_scale = use_scale

        self.error_fraction = {}

        self.name = NNClassifier.name

        self.all_data = X_train.todense()

        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler()
        self.all_data = self.sc.fit_transform(self.all_data)

        pickle.dump(self.all_data, open("/tmp/X.p", "w+b"))


    def run_cross_validation(self, train, train_target, folds):
        pass

    def train_predict_all(self, x, y):

        new_x = x.todense()

        new_x = self.sc.transform(new_x)

        def create_baseline():
            model = Sequential()

            #model.add(Dropout(0.2, input_shape=(new_x.shape[1],)))
            model.add(Dense(units=512, activation='relu', input_dim=new_x.shape[1]))#best
            #model.add(Dense(units=256, activation='relu', input_dim=new_x.shape[1]))
            # model.add(Dense(units=128, activation='relu'))
            model.add(Dense(units=1, activation='sigmoid'))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            return model

        self.model = KerasClassifier(build_fn=create_baseline, epochs=500, batch_size=5, verbose=1)
        self.model.fit(new_x, y)

        probability_prediction_all = self.model.predict_proba(self.all_data)

        if self.model.classes_[1] == True:
            probability_prediction = probability_prediction_all[:, 1]
        else:
            probability_prediction = probability_prediction_all[:, 0]
        class_prediction = probability_prediction > 0.5

        return probability_prediction, class_prediction