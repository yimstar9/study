
##임성구

# (1) 일별 국가별 코로나 발생자수와 사망자 수를 기준으로 전처리 하시오. 일부
# 국가는 지역별로 코로나 발생자수와 사망자 수가 분리되어 있으니 국가별로
# 집계하고 국가, 총발생자수, 총사망자수, 일평균 발생자수, 일평균 사망자수 리
# 스트를 제시하시오.
# (난이도: 4, 배점: 30점)
import pandas as pd
import os

pd.options.display.max_rows =30
pd.options.display.min_rows =20
pd.options.display.max_columns =10
pd.set_option('display.max.colwidth',15)


nDf=pd.DataFrame()
tDf=pd.DataFrame()
rs=pd.DataFrame()
final=pd.DataFrame()
filename=os.listdir('covid19daily')
df = pd.read_csv('12-31-2020.csv')
df.drop(labels=['FIPS','Admin2','Province_State','Last_Update','Long_','Lat','Recovered','Active','Combined_Key','Incident_Rate','Case_Fatality_Ratio'], axis=1)
tDf=df[['Country_Region', 'Confirmed', 'Deaths']].groupby('Country_Region').sum()

tDf.columns=[['12-31-2020','12-31-2020'],['Confirmed','Deaths']]
subDf=df[['Country_Region', 'Confirmed', 'Deaths']].groupby('Country_Region').sum()
#
# for i in range(len(filename)):
#     df = pd.read_csv('covid19daily/'+filename[i])
#     df.drop(labels=['FIPS','Admin2','Province_State','Last_Update','Long_','Lat','Recovered','Active','Combined_Key','Incident_Rate','Case_Fatality_Ratio'], axis=1)
#     df.loc[df.Confirmed.isnull(), 'Confirmed'] = 'NoData'
#     df.loc[df.Deaths.isnull(), 'Deaths'] = 'NoData'
#     tDf=df[['Country_Region', 'Confirmed', 'Deaths']].groupby('Country_Region').agg(['sum'])
#     tDf.columns.naume=[filename[i]]
#     tDf.columns=tDf.columns.droplevel(1)
#     tDf.columns.names = [filename[i]]
#     pd.concat([tDf, nDf], ignore_index=True)
#     # pd.merge(nDf,tDf,how='outer',left_index=True, right_index=True)

for i in range(len(filename)):
    df1= pd.read_csv('covid19daily/'+filename[i])
    df1.drop(labels=['FIPS','Admin2','Province_State','Last_Update','Long_','Lat','Recovered','Active','Combined_Key','Incident_Rate','Case_Fatality_Ratio'], axis=1)
    nDf=df1[['Country_Region', 'Confirmed', 'Deaths']].groupby('Country_Region').sum()
    nDf=nDf.sub(subDf)
    nDf.columns = [[filename[i][:-4],filename[i][:-4]],['Confirmed','Deaths']]
    result = pd.merge(tDf, nDf, how='outer', left_index=True, right_index=True)
    subDf = df1[['Country_Region', 'Confirmed', 'Deaths']].groupby('Country_Region').sum()
    tDf=result

rs=(result.drop(labels=['12-31-2020'],axis=1,level=0)).copy(deep=True)
print(rs)
rs.columns=rs.columns.droplevel(0)

final['총확진자']=rs['Confirmed'].sum(axis=1)
final['일평균 확진자']=rs['Confirmed'].mean(axis=1)
final['총사망자']=rs['Deaths'].sum(axis=1)
final['일평균 사망자']=rs['Deaths'].mean(axis=1)
print(final)
# result.to_csv('result.csv')

# (2) 데이터가 0인 경우(코로나 환자 0)와 데이터가 없는 경우를 구분하여 전처
# 리하고 전처리 시 data가 없는 국가는 제외하고 제외된 국가 리스트를 제시하
# 시오

Ans = final.loc[final['총확진자']==0]
print(Ans.index)

# (3) 2021년 1년동안 코로나 총 발생자수, 총 사망자수, 일평균 발생자수, 일평균
# 사망자 수를 기준으로 가장 많은 20개 국가를 내림차순으로 정렬하고 총 발생
# 자수, 총 사망자수, 일평균 발생자수, 일평균 사망자 수를 리포트 하시오. (4가
# 지 기준 각각 sorting)
# (난이도: 4, 배점: 30점)

totConfirmed20 = final.sort_values('총확진자',ascending=False).reset_index()
print(totConfirmed20.head(20))
meanConfirmed20 = final.sort_values('일평균 확진자',ascending=False).reset_index()
print(meanConfirmed20.head(20))
totDeaths20 = final.sort_values('총사망자',ascending=False).reset_index()
print(totDeaths20.head(20))
meanDeaths20 = final.sort_values('일평균 사망자',ascending=False).reset_index()
print(meanDeaths20.head(20))

# (4) 2021년 1년동안 대한민국에서 발생한 총 코로나 발생자수와 총 사망자 수
# 와 일평균 발생자수와 일평균 사망자 수를 리포트 하시오.
# (난이도: 3, 배점: 20점)

print(final.loc['Korea, South'])