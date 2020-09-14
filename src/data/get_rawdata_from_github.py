#import require packages
import subprocess
import os
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import json

# function to gather data from johns hopkins by doing git_pull
def get_johns_hopkins_data():
    git_pull = subprocess.Popen('git pull',
                         cwd = os.path.dirname('data/raw/COVID-19'),
                         shell = True,stdout = subprocess.PIPE,stderr = subprocess.PIPE )
    (out, error) = git_pull.communicate()
    print("Error : " + str(error))
    print("out : " + str(out))

# function to gather data of only germany z.B. RKI website: (this data will be not used in the project)
def get_germany_data():
    data_germany=requests.get('https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_Landkreisdaten/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json')

    json_object=json.loads(data_germany.content)
    final_list=[]
    for pos,each_dict in enumerate (json_object['features'][:]):
        final_list.append(each_dict['attributes'])
    pd_final_list=pd.DataFrame(final_list)
    pd_final_list.to_csv('data/raw/NPGEO/Germany_statewise_data.csv',sep=';')
    #print(' Number of regions rows: '+str(pd_full_list.shape[0]))

if __name__ == '__main__':
    get_johns_hopkins_data()
    get_germany_data()
