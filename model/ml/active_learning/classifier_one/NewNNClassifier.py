from sklearn.svm import SVC
from keras.models import Sequential


from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
import pickle

class NewNNClassifier(object):
    name = 'NN'

    def __init__(self, X_train, X_test, use_scale=True, feature_names=None):
        self.params = {}
        self.model = {}
        self.use_scale = use_scale

        self.error_fraction = {}

        self.name = NewNNClassifier.name

        print('before dense')
        all_data = X_train.todense()
        print('after dense')

        self.sc = StandardScaler()
        all_data = self.sc.fit_transform(all_data)

        print('after scaling')

        #pickle.dump(self.all_data, open("/tmp/X.p", "w+b"))


        print(feature_names)

        char_features_per_column_ids = {}
        correlation_features_per_column_ids = {}
        self.one_hot_column_ids = []
        self.metadata_ids = []

        for list_i in range(len(feature_names)):
            if '_letter_' in feature_names[list_i]:
                id = feature_names[list_i].find('_letter_')
                column_name = feature_names[list_i][0:id]
                if not column_name in char_features_per_column_ids:
                    char_features_per_column_ids[column_name] = []
                char_features_per_column_ids[column_name].append(list_i)
            elif '_word2vec_' in feature_names[list_i]:
                id = feature_names[list_i].find('_word2vec_')
                column_name = feature_names[list_i][0:id]
                if not column_name in correlation_features_per_column_ids:
                    correlation_features_per_column_ids[column_name] = []
                correlation_features_per_column_ids[column_name].append(list_i)
            elif 'column_id_' in feature_names[list_i]:
                self.one_hot_column_ids.append(list_i)
            else:
                self.metadata_ids.append(list_i)


        self.id_list = char_features_per_column_ids.values()
        self.id_list.extend(correlation_features_per_column_ids.values())

        print self.id_list

        self.all_input_matrices = []
        for my_list in self.id_list:
            self.all_input_matrices.append(all_data[:, my_list])
        self.all_input_matrices.append(all_data[:, self.one_hot_column_ids])
        self.all_input_matrices.append(all_data[:, self.metadata_ids])




    def run_cross_validation(self, train, train_target, folds):
        pass

    def train_predict_all(self, x, y):

        new_x = x.todense()
        new_x = self.sc.transform(new_x)



        def create_baseline():
            # define two sets of inputs
            inputs = []
            rep_models = []

            dimensionality_reduction = 1

            for my_list in self.id_list:
                current_input = Input(shape=(len(my_list),))
                inputs.append(current_input)
                current_output = Dense(8, activation="relu")(current_input)
                current_output = Dense(dimensionality_reduction, activation="relu")(current_output)
                current_output = Model(inputs=current_input, outputs=current_output)
                rep_models.append(current_output)

            # combine the output of the two branches
            combined_list = [m.output for m in rep_models]
            onehot_input = Input(shape=(len(self.one_hot_column_ids),))
            combined_list.append(onehot_input)
            metadata_input = Input(shape=(len(self.metadata_ids),))
            combined_list.append(metadata_input)
            combined = concatenate(combined_list)

            # apply a FC layer and then a regression prediction on the
            # combined outputs
            z = Dense(dimensionality_reduction * len(rep_models) + len(self.one_hot_column_ids) + len(self.metadata_ids), activation="relu")(combined)
            z = Dense(1, activation="sigmoid")(z)

            # our model will accept the inputs of the two branches and
            # then output a single value
            final_inputs = [m.input for m in rep_models]
            final_inputs.append(onehot_input)
            final_inputs.append(metadata_input)
            model = Model(inputs=final_inputs, outputs=z)

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])


            return model


        input_matrices = []
        for my_list in self.id_list:
            input_matrices.append(new_x[:, my_list])
        input_matrices.append(new_x[:, self.one_hot_column_ids])
        input_matrices.append(new_x[:, self.metadata_ids])

        self.model = KerasClassifier(build_fn=create_baseline, epochs=500, batch_size=5, verbose=1)
        self.model.fit(input_matrices, y)

        probability_prediction_all = self.model.predict_proba(self.all_input_matrices)

        if self.model.classes_[1] == True:
            probability_prediction = probability_prediction_all[:, 1]
        else:
            probability_prediction = probability_prediction_all[:, 0]
        class_prediction = probability_prediction > 0.5

        return probability_prediction, class_prediction