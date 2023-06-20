import pandas as pd
import numpy as np
import random as r
import re
from sklearn.preprocessing import StandardScaler
import pickle
## classes:
# Cleaner: for tabular data
# Visualizer: for exploratory analysis
class Cleaner:
	"""
	For automatic data cleaning
	inplace operation for the input data frame
	"""
	def __init__(self, df):
		assert isinstance(df, pd.DataFrame), "Input data must be a pandas DataFrame."
		self.df = df
	# def __repr__
	def rm_na_cols(self, na_percentage):
		assert 0 <= na_percentage <= 1, "na_percentage must be between 0 and 1"
		cnt = int(self.df.shape[0] * na_percentage)
		self.df.dropna(thresh = cnt, axis=1, inplace=True)
	def rm_no_outcome(self, nm=None):
		"""
		TODO: rm rows that don't have outcome value
		:param nm: name of outcome feature in dataset
		"""
		if nm:
			self.df.dropna(subset=[nm], inplace=True)
	def binary_to_int(self):
		"""
		TODO: turn binary variable to integer
		"""
		int_col = self.df.columns[self.df.apply(lambda x:len(x.dropna().unique()))==2].to_list() # find unique val=2 not including NaN
		self.df.loc[:,int_col] = self.df.loc[:,int_col].astype('Int64')
	def fix_colname(self):
		"""
		TODO: In column names, turn uncommon symbols to underline. Spaces will be replaced by underline
		"""
		new_nms = list(map(lambda col: re.sub(r'[^0-9a-zA-Z_]', '_', col).lower(), self.df.columns))
		self.df.columns = new_nms
		return self.df
	def auto_clean(self, na_percentage=0.99, nm=None):
		self.drop_duplicates(inplace=True) # rm duplicate rows
		self.rm_no_outcome(nm)             # drop rows with no outcome
		self.rm_na_cols(na_percentage)     # drop cols with missingness > na_percentage
		self.binary_to_int()               # turn binary variables to integer
		self.fix_colname()                 # rm weird symbols from column names
def overview(df):
	"""
	Check the data types & NA percentage & unique values of the data
	"""
	print(f'size: {df.shape}')
	output = df.apply(lambda x: (x.dtype,x.isna().mean(),len(x.unique()),x.unique()), axis=0).T
	output.columns = ["dtype", "NaN_percentage","unique_cnt","unique"]
	print(f'Data types:\n{output.dtype.value_counts()}')
	return output.sort_values(['NaN_percentage','unique_cnt'], ascending=False)
## Data preprocessing & feature extraction:
def extract_date_features(df, col, date_format='%Y-%m-%d', label=None):
	"""
	TODO: extract features from a date feature
	:param df: input data frame for inplace operation
	:param col: name of the date columns
	:param date_format: format of the date feature
	:param label: labeling for the extracted features
	"""
	if label is None:   # use the ori name as label
	    label = col
	df[col] = pd.to_datetime(df[col], format=date_format)
	df['yr_'+label] = df[col].dt.year
	df['mo_'+label] = df[col].dt.month
	df['day_'+label] = df[col].dt.day
	df['wday_'+label] = df[col].dt.weekday
def standardization(df, method='normalize', cols=None):
	"""
	TODO: Standardization for numeric features
	:param df: input data frame for inplace operation
	:param method: str(enum); must be one of ('normalize', 'min_max')
	:param cols: a list/idxes of specified columns for standardization; default is all numeric features
	"""
	methods = ('normalize', 'min_max')
	assert method in methods, f"Invalid value for 'method' parameter. Expected one of: {methods}"
	if cols is None:
	    cols = df.select_dtypes(include=np.number).columns
	if method == 'normalize':
	    df[cols] = df[cols].apply(lambda x:(x-x.mean())/x.std() )
	elif method == 'min_max':
	    df[cols] = df[cols].apply(lambda x:(x-x.min())/(x.max()-x.min()) )
def label_encoding(df, encode_cols=None, max_categories=30, as_category=False):
	"""
	TODO: turn categorical features into integer labeling
	:param df: data frame to do inplace operation
	:param encode_cols: a list/idxes of specified columns for encoding; default is all dtype='object' & 'str'
	:param max_categories: max unique categories within 1 feature to do label encoding
	:param as_category: turn labels to dtype=category if True
	:return: inplace operation to input df & return a table that matches origianl labeling & label encoding
	"""
	if encode_cols is None:
		encode_cols = df.select_dtypes(include='object').columns   # default features
	max_id = (df[encode_cols].apply(lambda x:len(x.unique())) < max_categories)
	encode_cols = encode_cols[max_id]   # rm the cols with unique count > max_categories
	# print(f"Features {encode_cols.tolist()} are transformed from [\'object\', \'string\'] to Int64 labels")
	excluded = max_id.index[~max_id].tolist()
	if len(excluded)>0:
		print(f"Varible {excluded} reached the max_categories, didn't do label encoding")
	check = (df[encode_cols].astype(str) != '-1').all()    # check '-1' not in features, otherwise can't be nan label
	assert check.all(), f"label -1 is in features: {check[check==False].index}"
	# keep the labels as a organized dataframe:
	labels = df[encode_cols].apply(lambda x: pd.Series(pd.factorize(x, sort=True)[1]))   # the origianl labeling
	df[encode_cols] = df[encode_cols].apply(lambda x: pd.factorize(x, sort=True)[0])     # encoded values
	df[encode_cols] = df[encode_cols].astype('Int64') # int64->Int64; to save nan without turning to float
	df[encode_cols] = df[encode_cols].replace([-1], np.nan)
	# if as_category:
	return labels

def match_cols(df_columns, pattern):
	"""
	TODO: keep the cols that matches regex patterns
	:param df: input data
	:param pattern: r""; re pattern for re.match()
	return: matched features idx
	"""
	matched_cols = df_columns[df_columns.str.match(pattern)]
	return matched_cols

def save_py(obj, location):
	"""
	TODO: to save a python object
	:param obj: object name
	:param location: location to save the file
	"""
	if location[-4:] != '.pkl': location += '.pkl'  # add file extension
	savefile = open(f'{location}', 'wb')
	pickle.dump(obj, savefile)
	savefile.close()
def load_py(location):
	file = open(f'{location}', 'rb')     # open a file, where you stored the pickled data
	data = pickle.load(file)
	file.close()
	return data







