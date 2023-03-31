import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import os
import glob

# separate X (features) and Y (label) for the dataset
def separate_X_Y(df: pd.DataFrame):
	"""This function is to separate the features and labels from a dataset

	Args:
			df (pd.DataFrame): processed data

	Returns:
			X: np 2d array, N*d
      Y: np 1d array, 1*N
	"""
	if 'class' in df.columns:
		df_class = df['class'].replace('AL', 'L').replace('BL', 'L')
		df_class = df_class.replace('AU', 'U').replace('BU', 'U')
		df_class = df_class.replace('U', 0).replace('L', 1)
		df_X = df.drop(['class', 'cand', 'ori_index'], axis=1)
		return df_X.to_numpy(), df_class.to_numpy()
	else:
		return df.to_numpy(), None
  
  
def build_model(classifier: RandomForestClassifier, df_train: pd.DataFrame, model_save_path: str):
  """this function is to train a random forest classifier and print our the trees and save the classifiers themselves

  Args:
      classifier (RandomForestClassifier): a random forest classifier
      df_train (pd.DataFrame): a random forest classifier
      model_save_path (str): a file path to save the trained model
  """
  
  # fit classifier with training data
  X_train, Y_train = separate_X_Y(df_train)
  classifier.fit(X_train, Y_train)
  # save classifier trained with pos-only data
  joblib.dump(classifier, model_save_path) 
  
  
def candidate_predict(df_test: pd.DataFrame, classifier_file_path: str) -> pd.DataFrame:
	"""This function is to predict the probability results

	Args:
			df_test (pd.DataFrame): test dataset
			classifier_file_path (str): the file path of trained classifier

	Returns:
			pd.DataFrame: probability ranking results of test dataset
	"""

	# separate X and Y for testing data
	X_test, _ = separate_X_Y(df_test)

	# get 'cand' for candidate predictions later
	cands = df_test['cand'].tolist()
	ori_index = df_test['ori_index'].tolist()

	df_pred = pd.DataFrame()
	# predict the probs for testing data
	# load the classifier
	classifier = joblib.load(classifier_file_path)
	# get predicted labels for each sample
	predicted = classifier.predict(X_test)
	# obtain probability of each sample
	probs = classifier.predict_proba(X_test)[:, 1]

	# build prediction results for samples
	df_pred = pd.DataFrame(predicted, columns=['predicted'])
	df_pred['prob'] = probs
	df_pred['cand'] = cands
	df_pred['ori_index'] = ori_index

	# build prediction results for candidates
	results = df_pred.groupby(['cand'], as_index=False).mean()
	results = results[['cand', 'prob']]
	results = results.sort_values(by='prob', ascending=False)

	return results

  
if __name__ == '__main__':
	random_state = 10
  
  # plant data (training data)
	fn_both_preprocessed_file_path = './data/plant_data/preprocessed_False_Negative_both.csv'
	df_fn = pd.read_csv(fn_both_preprocessed_file_path)
 
	# human data (test data)
	human_data_folder = './data/human_data/test_data/'

	# results save path
	model_save_path = './classifier.pkl'
  
	# define features
	selected_features = ['M1', 'M2', 'M3', 'zzM0', 'zzM1', 'zzM2', 'zzM3', 'MassDif_M1', 'MassDif_M2', 'class', 'cand', 'ori_index']

	# define random forest classifier
	rf_classifier = RandomForestClassifier(max_depth=5,
                           n_estimators=100,
                           random_state=random_state, 
                           min_samples_leaf=10)
	# build model
	build_model(rf_classifier, df_fn[selected_features], model_save_path)

	# create results folder
	if not os.path.exists('./results/'):
		os.makedirs('./results/')
	for preprocessed_test_file_path in glob.glob(human_data_folder + '*/preprocessed_*.csv'):
		df_test = pd.read_csv(preprocessed_test_file_path)
		df_test['class'] = 0
		df_prob_results = candidate_predict(df_test[selected_features], model_save_path)
		df_prob_results.to_csv('./results/results_' + os.path.basename(preprocessed_test_file_path), index=False)
