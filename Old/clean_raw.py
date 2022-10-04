import pandas as pd
import numpy as np
import networkx as nx

DATASET_FOLDER_PATH = 'dataset/'
IMPORT_FILE_NAME = 'telecomChurn.zip'
EXPORT_FILE_NAME = 'clean_data.csv'

def clean_raw():
	"""reads in raw dataset downloaded at relative path's dataset folder 
 	(No OS recognition, must activate anaconda in project folder) and 
  	outputs a cleaned version ready to be run by the model in the same folder 
   	as well.
	"""
	clean_data = pd.DataFrame()

	try:
		raw_data = pd.read_csv(DATASET_FOLDER_PATH+IMPORT_FILE_NAME)
	except:
		print(f'raw data {IMPORT_FILE_NAME} cannot be found in \
      			{DATASET_FOLDER_PATH}')
		return
	raw_data = raw_data.set_index('Customer_ID')
	raw_data.columns = map(str.lower,raw_data.columns)
	raw_data = raw_data[np.sort(raw_data.columns)]
 
	#drop columns that have weird codes where data descriptions are
	#no longer available and hence are no longer interpretable
	#We could OHE them and throw them in the model but since the end
	#goal is to create a dashboard where a user can input a customer profile,
	#uninterpretable variables cannot be entered by a human operator and hence 
	#should not be used
	raw_data = raw_data.drop(['ethnic','dwllsize','hhstatin'],axis=1)
	print('loaded data')

	#create an undirected graph of perfectly correlated missing columns
	#with each column as a node then group columns with perfectly correlated 
	#missingness together. This way we can treat multiple columns with
	#identical missingness simultaneously
	perf_miss_cols = get_perf_miss_cols(raw_data)

	#impute mode for the 2 rows with missing area data
	raw_data['area'] = raw_data['area'].fillna(raw_data['area'].mode().iloc[0])

	#get median values w.r.t area and month to impute missing values for
	#columns with missingness perfectly correlated to avg6qty
	avg6_cluster_mia_dct = (
		raw_data[
			(
				raw_data['months']
				==raw_data[raw_data['avg6qty'].isna()]['months'].mode().iloc[0]
			)
			&(~raw_data['avg6qty'].isna())
		].groupby('area')[list(perf_miss_cols[0])].median()
	)
	area_dct_imputation(avg6_cluster_mia_dct,clean_data,raw_data)

	#get median values w.r.t area and month to impute values for
	#columns with missingness perfectly correlated to change_rev 
	change_cluster_mia_dct = raw_data[
		(
			raw_data['months']
			==raw_data[raw_data['change_rev'].isna()]['months'].mode().iloc[0]
		)
		&(~raw_data['change_rev'].isna())
	].groupby('area')[list(perf_miss_cols[1])].median()
	area_dct_imputation(change_cluster_mia_dct,clean_data,raw_data)

	#get mode values to impute into missing household survey data
	for col in perf_miss_cols[2]:
		clean_data[col] = raw_data[col].fillna(raw_data[col].mode().iloc[0])

	#get median values w.r.t area only to impute values for
	#columns  with missingness perfectly correlated to rev_mean 
	mean_mia_cluster_dct = (
		raw_data[
			raw_data['months'] == raw_data[
				raw_data['rev_mean'].isna()
			]['months'].mode().iloc[0]
		].groupby('area')[list(perf_miss_cols[3])].median().to_dict()
	)
	area_dct_imputation(mean_mia_cluster_dct,clean_data,raw_data)

	#fill the last cluster of columns with perfectly correlated missingness
	#using the mode
	for col in perf_miss_cols[4]:
		clean_data[col] = raw_data[col].fillna(
			raw_data[col].mode().iloc[0]
		)

	clean_data['hnd_price'] = raw_data['hnd_price'].fillna(
		raw_data['hnd_price'].mode().iloc[0]
	)

	#Add remaining columns to clean_data
	remain_cols = list(
		set(raw_data.columns)
		-set(clean_data.columns)
	)
 
	clean_data[remain_cols] = raw_data[remain_cols]
	#number of cars missing implies no cars
	clean_data['numbcars'] = clean_data['numbcars'].fillna(0)
	print("handled missingness, treating last cols")
 
	#convert columns with boolean data to 0 and 1
	convert_binary_cols(clean_data)
	clean_marital(clean_data)
	clean_prizm_social(clean_data)
	
	#create a simplified column of whether household has children
	clean_data['has_kid'] = clean_data[
		clean_data.columns[clean_data.columns.str.contains('kid')]
	].apply(lambda row: row.any(),axis=1)
 
	#sort columns alphabetically
	clean_data = clean_data[np.sort(clean_data.columns)]

	clean_data.to_csv(DATASET_FOLDER_PATH+EXPORT_FILE_NAME)
	print(f'data exported to {DATASET_FOLDER_PATH+EXPORT_FILE_NAME}')

def get_perf_miss_cols(raw_data:pd.DataFrame)->list:
	perf_corr_cols = get_perf_mia_corrs(raw_data)
	G = nx.from_pandas_edgelist(
		perf_corr_cols.to_frame().reset_index(drop=True),
		0,
		1
	)
	perf_miss_cols = []
	#add each component to a list for easier access
	for c in nx.connected_components(G):
		sorted_c = list(c)
		sorted_c.sort()
		perf_miss_cols.append(sorted_c)
	perf_miss_cols.sort()
	return perf_miss_cols

def clean_marital(clean_data:pd.DataFrame):
    marital_dict = {
		'S':'single',
		'M':'married',
		'A':'separated',
		'B':'unmarried',
		'U':'unknown'
	}
    clean_data['marital'] = clean_data['marital'].map(marital_dict)
    
def clean_prizm_social(clean_data:pd.DataFrame):
    prizm_dict = {
		'S':'suburb',
		'U':'urban',
		'T':'town',
		'C':'city',
		'R':'unknown'
	}
    clean_data['prizm_social_one'] = \
        clean_data['prizm_social_one'].map(prizm_dict)

def get_perf_mia_corrs(raw_data:pd.DataFrame)->pd.Series:
	'''Get columns that hav perfectly correlated missingness with each other
	'''
	mia_corr = raw_data.isna().corr()
	perf_corr_cols = (
		mia_corr
		.where(np.triu(np.ones(mia_corr.shape),1).astype(bool))
		.where(mia_corr==1,np.nan)
		.stack().index
	)
	return perf_corr_cols

def convert_binary_cols(clean_data:pd.DataFrame):
	'''convert binary columns with binary dtypes to 1 or 0
	'''
	binary_str_map_dct = {
		'asl_flag':{'N':False,'Y':True},
		'creditcd':{'N':False,'Y':True},
		'dwlltype':{'S':False,'M':True},
		'forgntvl':{0:False,1:True},
  		'rv':{0:False,1:True},
		'truck':{0:False,1:True},
		'infobase':{'N':False,'M':True},
		'kid0_2':{'U':False,'Y':True},
		'kid3_5':{'U':False,'Y':True},
		'kid6_10':{'U':False,'Y':True},
		'kid11_15':{'U':False,'Y':True},
		'kid16_17':{'U':False,'Y':True},
		'ownrent':{'R':False,'O':True},
		'refurb_new':{'N':False,'R':True},
		'new_cell':{'U':np.nan,'Y':True,'N':False}
	}
	for key in binary_str_map_dct.keys():
		clean_data[key] = clean_data[key].map(binary_str_map_dct[key])

def area_dct_imputation(dct:dict,clean_data:pd.DataFrame,raw_data:pd.DataFrame):
	'''uses a dictionary to impute missing values 
	w.r.t area and adds it to clean data
	'''
	for key in dct.keys():
		clean_data[key] = raw_data[key].fillna(raw_data['area'].map(dct[key]))

if __name__ == "__main__":
    clean_raw()