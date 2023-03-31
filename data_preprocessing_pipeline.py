import pandas as pd
import numpy as np
from typing import List
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import glob
import os

# define the random state to keep the results reproducible
random_state = 10

# separate X (features) and Y (label) for the dataset
def separate_X_Y(df):
  '''
  :param df: df, processed data
  :return: X: np 2d array, N*d, d--feature dimentions
           Y: np 1d array, 1*N
  '''
  if 'class' in df.columns:
    df_class = df['class'].replace('AL', 'L').replace('BL', 'L')
    df_class = df_class.replace('AU', 'U').replace('BU', 'U')
    df_class = df_class.replace('U', 0).replace('L', 1)
    df_X = df.drop(['class', 'cand', 'ori_index'], axis=1)
    return df_X.to_numpy(), df_class.to_numpy()
  else:
    return df.to_numpy(), None


# data preprocessing function
def data_preprocessing_ratio(df_ori: pd.DataFrame):
  """preprocess the ratio data
  
  no return value, save the preprocessed ratio data to a csv file

  Args:
      df_ori (pd.DataFrame): filtered data
  """

  # define different groups od feature names
  drift_feature_names = [i for i in df_ori.columns if i.startswith('drift')]
  massdiff_feature_names = [
      i for i in df_ori.columns if i.startswith('MassDif')
  ]
  ratio_m_feature_names = ['M1', 'M2', 'M3']
  
  ## preprocessing steps
  # preprocess m values
  df_ratio = preprocess_m_values_ratio(df_ori, ratio_m_feature_names)
  df_ratio = impute_m_values(df_ratio, ['M1'])
  df_ratio = preprocess_m_values_discre_ratio(df_ratio, ['M2', 'M3'])
  # preprocess drift values
  df_ratio[drift_feature_names] = df_ratio[drift_feature_names].abs()
  # preprocess massdif values
  df_ratio[massdiff_feature_names] = df_ratio[massdiff_feature_names].fillna(100)

  return df_ratio

 
def preprocess_m_values_ratio(df_ori: pd.DataFrame,
                              ratio_m_feature_names: List[str]) -> pd.DataFrame:
  """preprocess the m_values feature set
  
  the final output contains "N/A", which should be imputed later
  
  Args:
      df_ori (pd.DataFrame): the dataframe with m_values unpreprocessed
      ratio_m_feature_names (List[str]): the m_value feature set that needs to 
        be preprocessed

  Returns:
      pd.DataFrame: a df with m_values preprocessed
  """
  
  # drop rows with "M0" as "N/A"
  df_ratio = df_ori[df_ori['M0'].notna()]

  # calculate the ratios
  for feature in ratio_m_feature_names:
    df_ratio[feature] = df_ratio[feature] / df_ratio['M0']

  # drop the "M0" features
  df_ratio.drop("M0", axis=1, inplace=True)

  return df_ratio


def impute_m_values(df_ratio: pd.DataFrame, ratio_m_feature_names: List[str]) -> pd.DataFrame:
  """impute m values for ratio data

  Args:
      df_ratio (pd.DataFrame): the dataframe with m_values unimputed
      ratio_m_feature_names (List[str]): the m_value feature set names that needs 
        to be imputed

  Returns:
      pd.DataFrame: a df with m_values imputed
  """
  
  # construct imputer
  rm_rg = RandomForestRegressor(max_depth=5, random_state=random_state)
  imputer_rm = IterativeImputer(estimator = rm_rg, tol = 1e-8, max_iter = 30, imputation_order = 'random', missing_values=np.nan, random_state=random_state)
  # obtain m_values for imputing
  X, _ = separate_X_Y(df_ratio[ratio_m_feature_names])
  # impute m_values
  X = imputer_rm.fit_transform(X)
  df_ratio[ratio_m_feature_names] = X
  
  return df_ratio


# this function must be invoked after "preprocess_m_values_ratio",
# it is based on the preprocessed ratio m_values
def preprocess_m_values_discre_ratio(
    df_ratio: pd.DataFrame, ratio_m_feature_names: List[str]) -> pd.DataFrame:
  """preprocess the m_values for discretized ratio data

  Args:
      df_ratio (pd.DataFrame): the ratio data with m_values undiscretized 
        (m_values contains N/A values)
      ratio_m_feature_names (List[str]): the m_value feature set that needs to 
        be preprocessed

  Returns:
      pd.DataFrame: a df with selected m_values being discretized
  """

  # fill the m_values with -100 to be prepared for discretization later
  df_ratio[ratio_m_feature_names] = df_ratio[ratio_m_feature_names].fillna(-100)
  # define categories for each m_value
  m_categories = [-100, 0, 10, 20, 30, 40, 50, 60, 70]
  # define bin percentile edges for each m_value
  m_bins = [0, 0.2, 0.32, 0.44, 0.56, 0.68, 0.8, 1]

  # discretize each m_value
  for feature in ratio_m_feature_names:
    # separate the N/A and 0 values from the remaining concrete values
    df_categorical_na_0 = df_ratio[df_ratio[feature] <= 0]
    df_categorical_remain = df_ratio[df_ratio[feature] > 0]
    # discretize the feature by the categories defined by m_bins_edge
    bin_series, bins_edges = pd.qcut(df_categorical_remain[feature],
                                    q=m_bins,
                                    labels=[10, 20, 30, 40, 50, 60, 70],
                                    retbins=True)
    # drop the original feature column
    df_categorical_remain.drop(feature, axis=1, inplace=True)
    # substitute the original data with the discretized data of this feature
    df_categorical_remain[feature] = bin_series
    # concatenate the df of N/A and 0 values with the df of discretized values
    df_ratio = pd.concat([df_categorical_na_0, df_categorical_remain])

  return df_ratio


# data filtering functions -- plant data (training data)
def filter_plant_candidates(df: pd.DataFrame) -> pd.DataFrame:
  """remove candidates with filtering conditions

  Args:
      df (pd.DataFrame): original df

  Returns:
      pd.DataFrame: df after filtering
  """
  
  print('total candidates: ' + str(len(df.cand.unique())))

  # filter out M0 == NA
  df_drop_m0 = df.loc[df['M0'].isna()]
  df.drop(df_drop_m0.index, inplace=True)
  print('samples removed by M0==NA: ' + str(len(df_drop_m0)))
  print('L samples removed by M0==NA: ' + str(len(df_drop_m0.loc[df_drop_m0['class'] == 'L'])))
  
  # filter out the samples where zzM0 >= 0.1
  df_drop_zzm0 = df.loc[df['zzM0'] > 0.1]
  df.drop(df_drop_zzm0.index, inplace=True)
  print('samples removed by zzM0>0.1: ' + str(len(df_drop_zzm0)))
  print('L samples removed by zzM0>0.1: ' + str(len(df_drop_zzm0.loc[df_drop_zzm0['class'] == 'L'])))

  # filter out the samples where M1/M0 > 10
  df_drop_m1 = df.loc[df['M1'] / df['M0'] > 10]
  df.drop(df_drop_m1.index, inplace=True)
  print('samples removed by M1/M0>10: ' + str(len(df_drop_m1)))
  print('L samples removed by M1/M0>10: ' + str(len(df_drop_m1.loc[df_drop_m1['class'] == 'L'])))

  # filter out the samples where xdrift > 10
  df_drop_xdrift = df.loc[df['xdrift'] > 10]
  df.drop(df_drop_xdrift.index, inplace=True)
  print('samples removed by xdrift>10: ' + str(len(df_drop_xdrift)))
  print('L samples removed by xdrift>10: ' + str(len(df_drop_xdrift.loc[df_drop_xdrift['class'] == 'L'])))

  # filter out candidates with only 1 sample
  df_before = df.copy()
  gp = df.groupby('cand')
  size = gp.size()
  cands = size[size > 1].index.tolist()
  df = df.loc[df['cand'].isin(cands)]
  print('candidates removed if only one sample left: ' + str(len(df_before.cand.unique()) - len(df.cand.unique())))
  df_L_before = df_before.loc[df_before['class'] == 'L']
  df_L_now = df.loc[df['class'] == 'L']
  print('L candidates removed if only one sample left: ' + str(len(df_L_before.cand.unique()) - len(df_L_now.cand.unique())))
  print('leftover candidates: ' + str(len(df.cand.unique())))
  print('L leftover candidates: ' + str(len(df_L_now.cand.unique())))
  print('~~~~~~~~~~~')
  
  return df


# data filtering functions -- human data (validation & test data)
def filter_human_candidates(df: pd.DataFrame) -> pd.DataFrame:
  
  """remove candidates with filtering conditions

  Args:
      df (pd.DataFrame): original df

  Returns:
      pd.DataFrame: df after filtering
  """
  
  print('total candidates: ' + str(len(df.cand.unique())))
  
  # filter out M0 == NA
  df_drop_m0 = df.loc[df['M0'].isna()]
  df.drop(df_drop_m0.index, inplace=True)
  print('samples removed by M0==NA: ' + str(len(df_drop_m0)))
  
  # filter out the samples where zzM0 >= 0.1
  df_drop_zzm0 = df.loc[df['zzM0'] > 0.1]
  df.drop(df_drop_zzm0.index, inplace=True)
  print('samples removed by zzM0>0.1: ' + str(len(df_drop_zzm0)))

  # filter out the samples where M1/M0 > 10
  df_drop_m1 = df.loc[df['M1'] / df['M0'] > 10]
  df.drop(df_drop_m1.index, inplace=True)
  print('samples removed by M1/M0>10: ' + str(len(df_drop_m1)))

  # filter out the samples where M2/M1 > 10
  df_drop_m2 = df.loc[df['M2'] / df['M1'] > 10]
  df.drop(df_drop_m2.index, inplace=True)
  print('samples removed by M2/M1>10: ' + str(len(df_drop_m2)))

  # filter out the samples where xdrift > 10
  df_drop_xdrift = df.loc[df['xdrift'] > 10]
  df.drop(df_drop_xdrift.index, inplace=True)
  print('samples removed by xdrift>10: ' + str(len(df_drop_xdrift)))
  
  # filter our the samples where abs(driftM0) > 10 
  df_drop_drift_M0 = df.loc[abs(df['driftM0']) > 10]
  df.drop(df_drop_drift_M0.index, inplace=True)
  print('samples removed by driftM0>10: ' + str(len(df_drop_drift_M0)))
  
  # filter our the samples where abs(driftM1) > 10 
  df_drop_drift_M1 = df.loc[abs(df['driftM1']) > 10]
  df.drop(df_drop_drift_M1.index, inplace=True)
  print('samples removed by driftM1>10: ' + str(len(df_drop_drift_M1)))

	# filter out candidates with only 1 sample
  cand_before = len(df.cand.unique())
  gp = df.groupby('cand')
  size = gp.size()
  cands = size[size > 2].index.tolist()
  df = df.loc[df['cand'].isin(cands)]
  print('candidates removed if only one sample left: ' + str(cand_before - len(df.cand.unique())))
  print('leftover candidates: ' + str(len(df.cand.unique())))
  print('~~~~~~~~~~~')
  
  return df


def plant_data_preprocessing(df_fn: pd.DataFrame) -> pd.DataFrame:
	"""the plant data (training data) preprocessing pipeline

	Args:
			df_fn (pd.DataFrame): the original dataset

	Returns:
			pd.DataFrame: a preprocessed df
	"""
 
	# combine the freature "cand" and "class" together, to have a unique "cand" to
	# group & abandon "class" to avoid ambiguity with "class_YB"
	df_fn['cand'] = df_fn['cand'] + '_' + df_fn['class']
	# change the feature "class" of fn both -- apply "class_YB" to "class"
	df_fn['class'] = df_fn['class_YB']
	df_fn.drop('class_YB', axis=1, inplace=True)
	# drop the features "plant" & "FN"
	df_fn.drop('plant', axis=1, inplace=True)
	df_fn.drop('FN', axis=1, inplace=True)
	# apply filter conditions for plant df
	df_fn = filter_plant_candidates(df_fn)
	df_ratio = data_preprocessing_ratio(df_fn)
	
	return df_ratio


def human_data_preprocessing(df_human: pd.DataFrame) -> List[pd.DataFrame]: 
	"""the human data (validation & test data) preprocessing pipeline

	Args:
			df_human (pd.DataFrame): the original dataset

	Returns:
			List[pd.DataFrame]: a list of preprocessed dfs, including both Labeled condition & Unlabeled condition
	"""
	# get samples where 'condition' == 'L' or 'condition' == 'U'
	df_human_L = df_human.loc[df_human['condition'] == 'L']
	df_human_U = df_human.loc[(df_human['condition'] == 'UL') | (df_human['condition'] == 'U')]

	# apply filtering conditions for all human data
	df_human_L = filter_human_candidates(df_human_L)
	df_human_U = filter_human_candidates(df_human_U)

	# apply data preprocessing for all human data
	df_human_L_ratio = data_preprocessing_ratio(df_human_L)
	df_human_U_ratio = data_preprocessing_ratio(df_human_U)
 
	return [df_human_L_ratio, df_human_U_ratio]


if __name__ == '__main__':
  # plant data file paths
	fn_both_file_path = './data/plant_data/FalseNegSetBoth_newzz.csv'
	fn_both_preprocessed_file_path = './data/plant_data/preprocessed_False_Negative_both.csv'

	# human test data folder path
	test_data_folder = './data/human_data/test_data/'

	# preprocess the plant dataset	
	df_fn = pd.read_csv(fn_both_file_path)
	df_ratio = plant_data_preprocessing(df_fn)
	# save as csv file
	df_ratio.to_csv(fn_both_preprocessed_file_path, index=False)
 
	# preprocess the human test datasets
	for test_data_file_path in glob.glob(test_data_folder + "*/*.csv"):
		# get dfs for all human data
		df_test = pd.read_csv(test_data_file_path)
		[df_test_L, df_test_U] = human_data_preprocessing(df_test)
		# save as csv file
		file_name = '/preprocessed_' + os.path.splitext(os.path.basename(test_data_file_path))[0]
		df_test_L.to_csv(os.path.dirname(test_data_file_path) + file_name + '_L.csv', index=False)
		df_test_U.to_csv(os.path.dirname(test_data_file_path) + file_name + '_U.csv', index=False)
