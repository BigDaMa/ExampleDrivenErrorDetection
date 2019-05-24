import pickle

from ml.active_learning.library import *
import sys
from sklearn.metrics import confusion_matrix
from ml.Word2VecFeatures.Word2VecFeatures import Word2VecFeatures
from ml.features.ActiveCleanFeatures import ActiveCleanFeatures
from ml.features.ValueCorrelationFeatures import ValueCorrelationFeatures
from ml.features.BoostCleanMetaFeatures import BoostCleanMetaFeatures
import operator
import pickle


def go_to_next_column_prob(diff_certainty):
	certainty_columns ={}

	for key in diff_certainty.keys():
		certainty_columns[key] = (np.sum(diff_certainty[key]) / len(diff_certainty[key])) * 2

	return min(certainty_columns.iteritems(), key=operator.itemgetter(1))[0]




def go_to_next_column_round(column_id, dataSet):
	column_id = column_id + 1
	if column_id == dataSet.shape[1]:
		column_id = 0
	return column_id

def go_to_next_column_random(dataSet):
	my_list = dataSet.get_applicable_columns()
	id = np.random.randint(len(my_list))
	return my_list[id]

def go_to_next_column(dataSet, statistics,
					  use_max_pred_change_column_selection,
					  use_max_error_column_selection,
					  use_min_certainty_column_selection,
					  use_random_column_selection):
	if use_min_certainty_column_selection:
		return go_to_next_column_prob(statistics['certainty'])
	if use_max_pred_change_column_selection:
		return max(statistics['change'].iteritems(), key=operator.itemgetter(1))[0]
	if use_max_error_column_selection:
		return min(statistics['cross_val_f'].iteritems(), key=operator.itemgetter(1))[0]
	if use_random_column_selection:
		return go_to_next_column_random(dataSet)


def load_model(dataSet, classifier):
	dataset_log_files = {}
	dataset_log_files[HospitalHoloClean().name] = "hospital"
	dataset_log_files[BlackOakDataSetUppercase().name] = "blackoak"
	dataset_log_files[FlightHoloClean().name] = "flight"
	# not yet
	dataset_log_files[Salary().name] = "hospital"  # be careful
	dataset_log_files[Book().name] = "hospital"  # be careful

	potential_model_dir = Config.get("column.potential.models")

	return pickle.load(
		open(potential_model_dir + "/model" + dataset_log_files[dataSet.name] + "_" + classifier.name + ".p"))


def add_lstm_features(data, use_lstm_only, all_matrix_train, feature_name_list):
	lstm_path = ""
	if data.name == 'Flight HoloClean':
		lstm_path = Config.get('lstm.folder') + "/Flights/last/"
	elif data.name == 'HospitalHoloClean':
		lstm_path = Config.get('lstm.folder') + "/HospitalHoloClean/last/"
	elif data.name == 'BlackOakUppercase':
		lstm_path = Config.get('lstm.folder') + "/BlackOakUppercase/last/"
	else:
		raise Exception('We have no potential model for this dataset yet')

	all_matrix_train_deep = read_compressed_deep_features(lstm_path)

	all_matrix_test = None
	feature_name_list_deep = ['deep ' + str(dfeature) for dfeature in range(all_matrix_train_deep.shape[1])]

	if use_lstm_only:
		all_matrix_train = all_matrix_train_deep
		feature_name_list = feature_name_list_deep
	else:
		all_matrix_train = hstack((all_matrix_train, all_matrix_train_deep)).tocsr()
		feature_name_list.extend(feature_name_list_deep)

	return all_matrix_train, all_matrix_test, feature_name_list


def augment_features_with_predictions(data_x_matrix, all_matrix_train, current_predictions, column_id, train_chosen_ids):
	x_all = all_matrix_train.copy()
	for column_number, last_predictions in current_predictions.iteritems():
		if column_number != column_id:
			select_predictions = np.matrix(last_predictions).transpose()
			try:
				data_x_matrix = hstack((data_x_matrix, select_predictions[train_chosen_ids[column_id], :])).tocsr()
				x_all = hstack((x_all, select_predictions)).tocsr()
			except:
				data_x_matrix = np.hstack((data_x_matrix, select_predictions[train_chosen_ids[column_id], :]))
				x_all= np.hstack((x_all, select_predictions))
	return data_x_matrix, x_all


def augment_features_with_predictions_test(all_matrix_test, current_predictions_test, column_id):
	x_all = all_matrix_test.copy()
	for column_number, last_predictions in current_predictions_test.iteritems():
		if column_number != column_id:
			select_predictions = np.matrix(last_predictions).transpose()
			try:
				x_all = hstack((x_all, select_predictions)).tocsr()
			except:
				x_all= np.hstack((x_all, select_predictions))
	return x_all



def run_multi( params):
	return run(**params)



def create_user_start_data(target, num_errors=2):
    error_ids = np.where(target == True)[0]
    correct_ids = np.where(target == False)[0]

    if (len(error_ids) < 2 or len(correct_ids) < 2):
        return [], []

    error_select_ids = range(len(error_ids))
    np.random.shuffle(error_select_ids)
    error_select_ids = error_select_ids[0:num_errors]

    correct_select_ids = range(len(correct_ids))
    np.random.shuffle(correct_select_ids)
    correct_select_ids = correct_select_ids[0:num_errors]

    list_ids = []
    list_ids.extend(error_ids[error_select_ids])
    list_ids.extend(correct_ids[correct_select_ids])

    return target[list_ids], list_ids


def create_next_part(y_pred, batchsize):
    diff = np.absolute(y_pred - 0.5)
    sorted_ids = np.argsort(diff)

    if np.sum(diff) == 0.0 or np.sum(diff) == len(diff) * 0.5:
         np.random.shuffle(sorted_ids)

    print(diff[sorted_ids][0:100])

    return sorted_ids[0:batchsize]

def create_unique_next_part(y_pred, batchsize, dataset):
    diff = np.absolute(y_pred - 0.5)
    sorted_ids = np.argsort(diff)

    if np.sum(diff) == 0.0 or np.sum(diff) == len(diff) * 0.5:
         np.random.shuffle(sorted_ids)

    print(diff[sorted_ids][0:batchsize])

    new_ids = []
    id_counter = 0
    unique_set = set()
    while len(new_ids) < batchsize and id_counter < len(sorted_ids):
        n_id = sorted_ids[id_counter]
        row_id = n_id % dataset.shape[0]
        col_id = (n_id - row_id) / dataset.shape[0]
        my_key = (dataset.values[row_id, col_id], col_id)

        if not my_key in unique_set:
            unique_set.add(my_key)
            new_ids.append(n_id)

        id_counter += 1
    return new_ids

def generate_html(data, detection_result):
    my_html = '<table style="width:100%">'

    my_html += '<tr>'
    for col_i in range(data.shape[1]):
        my_html += '<th>' + str(data.clean_pd.columns[col_i]) +'</th>'
    my_html += '</tr>'

    for row_i in range(data.shape[0]):
        my_html += '<tr>'
        for col_i in range(data.shape[1]):
            cell_color = '#ffffff'
            if data.matrix_is_error[row_i, col_i] and detection_result[col_i*data.shape[0] + row_i]:
                cell_color = '#32CD32'
            if data.matrix_is_error[row_i, col_i] and not detection_result[col_i*data.shape[0] + row_i]:
                cell_color = '#ff0000'
            if not data.matrix_is_error[row_i, col_i] and detection_result[col_i*data.shape[0] + row_i]:
                cell_color = '#800080'

            my_html += '<td bgcolor="' + cell_color + '">' + str(data.dirty_pd.values[row_i, col_i]) + '</td>'
        my_html += '</tr>'
    my_html += '</table>'

    text_file = open("/tmp/current_result" + str(time.time()) + ".html", "w")
    text_file.write(my_html)
    text_file.close()






def run(dataSet,
			 classifier_model,
			 number_of_round_robin_rounds=2,
			 train_fraction=1.0,
			 ngrams=1,
			 runSVD=False,
			 is_word=False,
			 use_metadata = True,
			 use_metadata_only = False,
			 use_lstm=False,
             use_lstm_only=False,
			 user_error_probability=0.00,
			 step_size=10,
			 cross_validation_rounds=1,
			 checkN=10,
			 label_iterations=6,
			 run_round_robin=False,
			 correlationFeatures=True,
			 use_tf_idf=True,
			 use_word2vec=False,
			 use_word2vec_only=False,
			 w2v_size=100,
			 use_active_clean=False,
			 use_activeclean_only=False,
			 use_cond_prob=False,
			 use_cond_prob_only=False,
		     use_boostclean_metadata=False,
		     use_boostclean_metadata_only=False,
             store_results=False,
		     use_random_column_selection=False,
		     use_max_pred_change_column_selection=False,
             use_max_error_column_selection=False,
			 use_min_certainty_column_selection=True,
             visualize_models=False,
		     output_detection_result=0
			 ):

	start_time = time.time()


	all_fscore = []
	all_precision = []
	all_recall = []

	all_fscore_test = []
	all_precision_test = []
	all_recall_test = []

	save_fscore = []
	save_precision = []
	save_recall = []

	save_fscore_test = []
	save_precision_test = []
	save_recall_test = []

	save_labels = []
	save_certainty = []
	save_time = []


	all_time = []

	use_change_features = True

	if run_round_robin:
		number_of_round_robin_rounds = 10000
		label_iterations = 71
		checkN = 10

	feature_names_potential = ['distinct_values_fraction', 'labels', 'certainty', 'certainty_stddev', 'minimum_certainty']

	for i in range(100):
		feature_names_potential.append('certainty_histogram' + str(i))

	feature_names_potential.append('predicted_error_fraction')

	for i in range(7):
		feature_names_potential.append('icross_val' + str(i))

	feature_names_potential.append('mean_cross_val')
	feature_names_potential.append('stddev_cross_val')

	feature_names_potential.append('training_error_fraction')

	for i in range(100):
		feature_names_potential.append('change_histogram' + str(i))

	feature_names_potential.append('mean_squared_certainty_change')
	feature_names_potential.append('stddev_squared_certainty_change')

	for i in range(10):
		feature_names_potential.append('batch_certainty_' + str(i))

	if use_change_features:
		feature_names_potential.append('no_change_0')
		feature_names_potential.append('no_change_1')
		feature_names_potential.append('change_0_to_1')
		feature_names_potential.append('change_1_to_0')

	print(str(feature_names_potential))

	size = len(feature_names_potential)
	for s in range(size):
		feature_names_potential.append(feature_names_potential[s] + "_old")

	which_features_to_use = []
	for feature_index in range(len(feature_names_potential)):
		if True: #not 'histogram' in feature_names_potential[feature_index]:
			which_features_to_use.append(feature_index)
	print which_features_to_use

	feature_names_potential = [i for j, i in enumerate(feature_names_potential) if j in which_features_to_use]

	feature_gen_time = 0.0

	for check_this in range(checkN):

		ts = time.time()
		f = open(Config.get("logging.folder") + "/logging_output/label_log_progress_" + dataSet.name + "_" + str(check_this) + "_" +  str(ts) + ".csv",
				 'w+')

		train_indices, test_indices = split_data_indices(dataSet, train_fraction, fold_number=check_this)

		total_start_time = time.time()

		feature_gen_start = time.time()

		all_matrix_train, all_matrix_test, feature_name_list = create_features(dataSet, train_indices, test_indices, ngrams,
																			   runSVD, is_word, use_tf_idf)

		print(all_matrix_train.shape)
		print("test" + str(len(feature_name_list)))



		if use_metadata:
			all_matrix_train, all_matrix_test, feature_name_list = add_metadata_features(dataSet, train_indices,
																						 test_indices, all_matrix_train,
																						 all_matrix_test, feature_name_list,
																						 use_metadata_only)

		if use_lstm:
			all_matrix_train, all_matrix_test, feature_name_list = add_lstm_features(dataSet, use_lstm_only, all_matrix_train,
																					 feature_name_list)



		if use_word2vec:
			w2v_features = Word2VecFeatures(vector_size=w2v_size)
			all_matrix_train, all_matrix_test, feature_name_list = w2v_features.add_word2vec_features(dataSet, train_indices,
																						 test_indices,
																						 all_matrix_train,
																						 all_matrix_test,
																						 feature_name_list,
																						 use_word2vec_only)



		if use_active_clean:
			ac_features = ActiveCleanFeatures()  # active clean
			all_matrix_train, all_matrix_test, feature_name_list = ac_features.add_features(dataSet,
																									  train_indices,
																									  test_indices,
																									  all_matrix_train,
																									  all_matrix_test,
																									  feature_name_list,
																									  use_activeclean_only)

		if use_cond_prob:
			ac_features = ValueCorrelationFeatures()
			all_matrix_train, all_matrix_test, feature_name_list = ac_features.add_features(dataSet,
																							train_indices,
																							test_indices,
																							all_matrix_train,
																							all_matrix_test,
																							feature_name_list,
																							use_cond_prob_only)


		if use_boostclean_metadata:
			ac_features = BoostCleanMetaFeatures()  # boost clean metadatda
			all_matrix_train, all_matrix_test, feature_name_list = ac_features.add_features(dataSet,
																									  train_indices,
																									  test_indices,
																									  all_matrix_train,
																									  all_matrix_test,
																									  feature_name_list,
																							          use_boostclean_metadata_only)


		print("features: %s seconds ---" % (time.time() - start_time))

		try:
			feature_matrix = all_matrix_train.tocsr()
		except:
			feature_matrix = all_matrix_train

		feature_matrix_per_column = []
		#todo: maybe create features only for erroneous columns
		debugging_ids = []
		for column_i in range(dataSet.shape[1]):
			one_hot_part = np.zeros(dataSet.shape)
			one_hot_part[:, column_i] = 1

			feature_matrix_per_column.append(hstack((feature_matrix, one_hot_part)).tocsr())
			feature_name_list.append('?column_id_' + str(column_i) + '_' + dataSet.clean_pd.columns[column_i])

			for row in range(dataSet.shape[0]):
				debugging_ids.append((row, column_i))

		all_columns_feature_matrix = vstack(feature_matrix_per_column)
		print(all_columns_feature_matrix.shape)

		print('features shape: ' + str(len(feature_name_list)))

		#find 4 labels for each erroneous column

		labeled_data = None
		labeled_target = None

		for column_i in range(dataSet.shape[1]):
			target_column = dataSet.matrix_is_error[train_indices, column_i]
			target_start, data_ids = create_user_start_data(target_column)

			onehot_ids = np.array(data_ids) + (column_i * dataSet.shape[0])

			#debug
			for dd in data_ids:
				new_oo_id = dd + (column_i * dataSet.shape[0])
				assert debugging_ids[new_oo_id] == (dd, column_i)

			if type(labeled_data) == type(None):
				labeled_data = all_columns_feature_matrix[onehot_ids]
				labeled_target = target_start
			else:
				labeled_data = vstack((labeled_data, all_columns_feature_matrix[onehot_ids]))
				labeled_target = np.concatenate((labeled_target, target_start))

		print(labeled_data.shape)
		print(labeled_target.shape)

		#train first classifier

		ground_truth_array = dataSet.matrix_is_error[train_indices, 0]
		for column_i in range(1, dataSet.shape[1]):
			ground_truth_array = np.concatenate((ground_truth_array, dataSet.matrix_is_error[train_indices, column_i]))

		print(ground_truth_array)
		assert len(ground_truth_array) == all_columns_feature_matrix.shape[0]

		#pickle.dump(ground_truth_array, open("/tmp/y.p", "w+b"))
		classifier = classifier_model(all_columns_feature_matrix, None, feature_names=feature_name_list)

		for run in range(label_iterations):

			if run <= 1:
				classifier.run_cross_validation(labeled_data, labeled_target, 7)
			save_labels.append(len(labeled_target))

			probability_prediction, class_prediction = classifier.train_predict_all(labeled_data, labeled_target)


			print(class_prediction[0:200])
			print(ground_truth_array[0:200])

			print("labels: " + str(len(labeled_target)))
			print('f1: ' + str(f1_score(ground_truth_array, class_prediction)))
			print('precision: ' + str(precision_score(ground_truth_array, class_prediction)))
			print('recall: ' + str(recall_score(ground_truth_array, class_prediction)))
			save_fscore.append(f1_score(ground_truth_array, class_prediction))
			save_precision.append(precision_score(ground_truth_array, class_prediction))
			save_recall.append(recall_score(ground_truth_array, class_prediction))

			#generate_html(dataSet, class_prediction)

			#per column score
			for col_i in range(dataSet.shape[1]):
				print("col " + str(col_i) + ' ' + str(dataSet.clean_pd.columns[col_i]))
				print('\tf1: ' + str(f1_score(ground_truth_array[col_i*dataSet.shape[0]: (col_i+1)*dataSet.shape[0]], class_prediction[col_i*dataSet.shape[0]: (col_i+1)*dataSet.shape[0]])))
				print('\tprecision: ' + str(precision_score(ground_truth_array[col_i*dataSet.shape[0]: (col_i+1)*dataSet.shape[0]], class_prediction[col_i*dataSet.shape[0]: (col_i+1)*dataSet.shape[0]])))
				print('\trecall: ' + str(recall_score(ground_truth_array[col_i*dataSet.shape[0]: (col_i+1)*dataSet.shape[0]], class_prediction[col_i*dataSet.shape[0]: (col_i+1)*dataSet.shape[0]])))
				print('\n')


			current_runtime = (time.time() - total_start_time)
			save_time.append(current_runtime)

			#new_ids = create_next_part(probability_prediction, step_size)
			new_ids = create_unique_next_part(probability_prediction, step_size, dataSet.dirty_pd)
			new_labels = []
			for n_id in new_ids:
				row_id = n_id % dataSet.shape[0]
				col_id = (n_id - row_id) / dataSet.shape[0]

				assert (row_id, col_id) == debugging_ids[n_id]
				print(str(dataSet.dirty_pd.values[row_id, col_id]) + 'was labelled as ' + str(dataSet.matrix_is_error[row_id, col_id]))
				new_labels.append(dataSet.matrix_is_error[row_id, col_id])

			labeled_data = vstack((labeled_data, all_columns_feature_matrix[new_ids]))
			labeled_target = np.concatenate((labeled_target, new_labels))

		all_fscore.append(save_fscore)
		all_precision.append(save_precision)
		all_recall.append(save_recall)

		all_fscore_test.append(save_fscore_test)
		all_precision_test.append(save_precision_test)
		all_recall_test.append(save_recall_test)

		all_time.append(save_time)

		return_dict = {}
		return_dict['labels'] = save_labels

		return_dict['fscore'] = all_fscore
		return_dict['precision'] = all_precision
		return_dict['recall'] = all_recall

		return_dict['fscore_test'] = all_fscore_test
		return_dict['precision_test'] = all_precision_test
		return_dict['recall_test'] = all_recall_test

		return_dict['time'] = all_time

		return return_dict






