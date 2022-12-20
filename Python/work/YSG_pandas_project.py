import pandas as pd
import os

pd.options.display.max_rows =30
pd.options.display.min_rows =20
pd.options.display.max_columns =10

nDf=pd.DataFrame()
filename=os.listdir('covid19daily')

df = pd.read_csv('12-31-2020.csv')
df1= pd.read_csv('covid19daily/12-31-2021.csv')
df.drop(labels=['FIPS','Admin2','Province_State','Last_Update','Long_','Lat','Recovered','Active','Combined_Key','Incident_Rate','Case_Fatality_Ratio'], axis=1)
df1.drop(labels=['FIPS','Admin2','Province_State','Last_Update','Long_','Lat','Recovered','Active','Combined_Key','Incident_Rate','Case_Fatality_Ratio'], axis=1)
tDf=df[['Country_Region', 'Confirmed', 'Deaths']].groupby('Country_Region').sum()
nDf=df1[['Country_Region', 'Confirmed', 'Deaths']].groupby('Country_Region').sum()
nDf=nDf.sub(tDf,fill_value=0)
tDf.columns=[['12-31-2020','12-31-2020'],['Confirmed','Deaths']]
nDf.columns = [['12-31-2021','12-31-2021'],['Confirmed','Deaths']]

result=pd.merge(tDf,nDf,how='outer',left_index=True, right_index=True)
result
result.to_csv('2021decribe.csv')