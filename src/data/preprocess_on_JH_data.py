#importing required packages
import pandas as pd
import numpy as np
from datetime import datetime

# function for store relational dataframe of Johns Hopkins data
def str_rel_df_JH_data():
    data_path='data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    pd_raw=pd.read_csv(data_path)
    pd_data=pd_raw.rename(columns={'Country/Region':'country','Province/State':'state'})
    pd_data['state']=pd_data['state'].fillna('no')
    pd_data=pd_data.drop(['Lat','Long'],axis=1)
    Final_relational_model=pd_data.set_index(['state','country']) \
                                .T                              \
                                .stack(level=[0,1])             \
                                .reset_index()                  \
                                .rename(columns={'level_0':'date',
                                                   0:'confirmed'},
                                                  )

    Final_relational_model['date']=Final_relational_model.date.astype('datetime64[ns]')
    Final_relational_model.to_csv('data/processed/COVID_relational_confirmed.csv',sep=';',index=False)
    print(' Total number of stored rows are: '+str(Final_relational_model.shape[0]))

if __name__ == '__main__':
    str_rel_df_JH_data()
