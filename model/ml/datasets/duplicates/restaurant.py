import pickle
import pandas as pd



restaurant_data = pickle.load( open( "/home/felix/SequentialPatternErrorDetection/DQM_datasets/restaurant/responses.p", "rb" ) )

print restaurant_data

ground_truth = pd.read_csv("/home/felix/SequentialPatternErrorDetection/DQM_datasets/restaurant/restaurant.csv", header=None)

true_matrix = ground_truth.values

#for pair in restaurant_data:
