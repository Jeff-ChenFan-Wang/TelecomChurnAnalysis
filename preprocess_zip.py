import pandas as pd
import numpy as np

DATASET_FOLDER_PATH = 'dataset/'
IMPORT_FILE_NAME = 'telecomChurn.zip'
EXPORT_FILE_NAME = 'clean_data.csv'

def clean_raw():
    """reads in raw dataset downloaded at relative path's dataset folder 
    (No OS recognition, must activate anaconda in project folder) and 
    outputs a cleaned version ready to be run by the model in the same folder 
    as well.
    """
    try:
        raw_data = pd.read_csv(
            DATASET_FOLDER_PATH+IMPORT_FILE_NAME,
            index_col='customerID'
        )
    except:
        print(f'raw data {IMPORT_FILE_NAME} cannot be found in \
                {DATASET_FOLDER_PATH}')
        return
    
    # Certain columns have a "No ..." string since they're 
    # sub categories of other columns hence we encode them as 0
        #(e.g If user doesn't have InternetService, Movie column shows up 
        # as "No InternetService" which in essence also means no Movies)
    raw_data = raw_data.replace('^No .+','No',regex=True)
    
    #Get list of boolean columns that need to be processed
        #We elect to keep gender as a str in case we need to account for 
        #non binary genders in the future
    two_val_cols = raw_data.columns[raw_data.nunique()==2].drop('gender')
    
    #Turn Boolean values into machine readable boolean values
    treated_bin_cols = (
        raw_data[two_val_cols]
        .replace('^Yes$',True,regex=True)
        .replace('^No$',False,regex=True)
    ).astype(bool)
    
    #join treated boolean columns with rest of data
    clean_data = (
        raw_data.drop(two_val_cols,axis=1)
        .join(treated_bin_cols)
        .copy()
    )
    
    #Turn SeniorCitizen that's encoded as 1 and 0 into booleans
    clean_data['SeniorCitizen'] = clean_data['SeniorCitizen'].astype(bool) 
    
    #Turn Total Charges into float dtype
    clean_data['TotalCharges'] = (
        clean_data['TotalCharges']
        .replace(' ',0)
        .astype(float)
    )
    

    
    clean_data.to_csv(DATASET_FOLDER_PATH+EXPORT_FILE_NAME)
    print(f'data exported to {DATASET_FOLDER_PATH+EXPORT_FILE_NAME}')

if __name__ == "__main__":
    clean_raw()