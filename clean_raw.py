import pandas as pd
import numpy as np
import scipy
import re
import networkx as nx

DATASET_FOLDER_PATH = 'dataset/'

def clean_raw():
	clean_data = pd.DataFrame()

	raw_data = pd.read_csv(DATASET_FOLDER_PATH+'telecomChurn.zip')
	raw_data = raw_data.set_index('Customer_ID')
	raw_data.columns = map(str.lower,raw_data.columns)
    raw_data = raw_data[np.sort(raw_data.columns)]

	#create an undirected graph of perfectly correlated missing columns
	#with each column as a node then group columns with perfectly correlated missingness together
	#this way we can treat multiple columns with identical missingness simultaneously
	perf_corr_cols = get_perf_mia_corrs(raw_data)
	G = nx.from_pandas_edgelist(
		perf_corr_cols.to_frame().reset_index(drop=True),
		0,
		1
	)
	perf_miss_cols = []
	#add each component to a list for easier access
	for c in nx.connected_components(G):
		perf_miss_cols.append(c)

	perf_miss_cols = np.sort(perf_miss_cols)

	#get median values w.r.t area and month then impute missing values for
	#columns related to avg6 month data using said dict
	avg6_cluster_mia_dct = (
		raw_data[
		(
			raw_data['months']
			==raw_data[raw_data['avg6qty'].isna()]['months'].mode().iloc[0]
		)
		&(~raw_data['avg6qty'].isna())
	].groupby('area')[list(perf_miss_cols[0])].median()
	area_dct_imputation(avg6_cluster_mia_dct,clean_data,raw_data)

	#get median values w.r.t area and month then impute values for
	#columns related to change_rev using said dict
	change_cluster_mia_dct = raw_data[
		(
			raw_data['months']
			==raw_data[raw_data['change_rev'].isna()]['months'].mode().iloc[0]
		)
		&(~raw_data['change_rev'].isna())
	].groupby('area')[list(perf_miss_cols[1])].median()
	area_dct_imputation(change_cluster_mia_dct,clean_data,raw_data)

	#get median values w.r.t area only then impute values for
	#columns related to mean values for months using said dict
	mean_mia_cluster_dct = (
		raw_data[
			raw_data['months'] == raw_data[
				raw_data['rev_mean'].isna()
			]['months'].mode().iloc[0]
		].groupby('area')[list(perf_miss_cols[3])].median().to_dict()
	)
	area_dct_imputation(mean_mia_cluster_dct,clean_data,raw_data)

	#zero fill remaining col's missing data
	for col in perf_miss_cols[4]:
		if raw_data.dtypes[col]=='O':
			clean_data[col] = raw_data[col].fillna('N')
		else:
			clean_data[col] = raw_data[col].fillna(0)

	clean_data[['dualband','refurb_new']] = (
		clean_data[['dualband','refurb_new']]
		.applymap(lambda x: 1 if x=='Y' else 0)
	)

	clean_data['hnd_price'] = (
		raw_data['hnd_price']
		.fillna(raw_data['hnd_price']
		.mode().iloc[0])
	)

	#handle remaining columns
	clean_data[cat_str_cols] = raw_data[cat_str_cols]
	remain_cols = list(
		set(raw_data.columns)
		-set(pd.concat([clean_data,raw_data[cat_str_cols]]).columns)
	)
	clean_data[remain_cols] = raw_data[remain_cols]
	clean_data['numbcars'] = clean_data['numbcars'].fillna(0)
	clean_data.columns = clean_data.columns.sort_values()

	pd.to_csv(DATASET_FOLDER_PATH+'clean_data.csv')

def convert_binary_cols(raw_data:pd.DataFrame, clean_data:pd.DataFrame):
	'''convert binary columns with binary dtypes to 1 or 0
	'''
	string_cols = raw_data.select_dtypes(include='object').columns
	num_cols = raw_data.select_dtypes(include=np.number).columns
	str_col_nuniq = raw_data[string_cols].nunique()
	binary_str_cols = list(str_col_nuniq[str_col_nuniq==2].index)
	binary_str_cols.append('new_cell')
	cat_str_cols = list(set(str_col_nuniq[str_col_nuniq>2].index)-{'new_cell'})

	binary_str_map_dct = {
		'asl_flag':{'N':0,'Y':1},
		'creditcd':{'N':0,'Y':1},
		'dwlltype':{'S':0,'M':1},
		'infobase':{'N':0,'M':1},
		'kid0_2':{'U':0,'Y':1},
		'kid3_5':{'U':0,'Y':1},
		'kid6_10':{'U':0,'Y':1},
		'kid11_15':{'U':0,'Y':1},
		'kid16_17':{'U':0,'Y':1},
		'ownrent':{'O':1,'R':0},
		'refurb_new':{'N':0,'R':1},
		'new_cell':{'U':np.nan,'Y':1,'N':0}
	}
	for key in binary_str_map_dct.keys():
		clean_data[key] = raw_data[key].map(binary_str_map_dct[key])


def get_perf_mia_corrs(raw_data:pd.DataFrame)->pd.MultiIndex:
	'''Get columns that hav perfectly correlated missingness with each other
	''''
	percentage_na = (
		raw_data.isna().sum().sort_values(ascending=False)/raw_data.shape[0]
	)
	mia_corr = raw_data.isna().corr()
	return perf_corr_cols = (
		mia_corr
		.where(np.triu(np.ones(mia_corr.shape),1).astype(bool))
		.where(mia_corr==1,np.nan)
		.stack().index
	)

def area_dct_imputation(dct:dict,clean_data:pd.DataFrame,raw_data:pd.DataFrame):
	'''uses a dictionary to impute missing values 
	w.r.t area and adds it to clean data
	'''
    for key in dct.keys():
        clean_data[key] = raw_data[key].fillna(raw_data['area'].map(dct[key]))

if __name__ == "__main__":
    clean_raw()