import pandas as pd

pd.options.display.max_rows =30
pd.options.display.min_rows =15
pd.options.display.max_columns =10

DataUrl = 'https://raw.githubusercontent.com/Datamanim/pandas/main/chipo.csv'
df = pd.read_csv(DataUrl)
df
df.loc[df['quantity']==3].head(5)
df.loc[df.quantity==3].head().reset_index()
df['new_price'] = df['item_price'].str[1:].astype('float')
Ans = df['new_price'].head()
Ans
len(df.loc[df['new_price'] <= 5])
#26
df.loc[df['item_name']=='Chicken Salad Bowl'].reset_index(drop=True)
#27 new_price값이 9 이하이고 item_name 값이 Chicken Salad Bowl 인 데이터 프레임을 추출하라
df.loc[(df['new_price']<=9)&(df['item_name']=='Chicken Salad Bowl')].head()
#28 df의 new_price 컬럼 값에 따라 오름차순으로 정리하고 index를 초기화 하여라
df.sort_values('new_price').reset_index(drop=True)
#29 df의 item_name 컬럼 값중 Chips 포함하는 경우의 데이터를 출력하라
df.loc[df.item_name.str.contains('Chips')].head()
#30 df의 짝수번째 컬럼만을 포함하는 데이터프레임을 출력하라
df.iloc[:,::2].head()
#31 df의 new_price 컬럼 값에 따라 내림차순으로 정리하고 index를 초기화 하여라
df.sort_values('new_price',ascending=False).reset_index(drop=True)
#32 df의 item_name 컬럼 값이 Steak Salad 또는 Bowl 인 데이터를 인덱싱하라
df.loc[(df['item_name'] =='Steak Salad')|(df['item_name'] == 'Bowl')]
#33 df의 item_name 컬럼 값이 Steak Salad 또는 Bowl 인 데이터를 데이터 프레임화 한 후, item_name를 기준으로 중복행이 있으면 제거하되 첫번째 케이스만 남겨라
ans=df.loc[(df['item_name'] =='Steak Salad')|(df['item_name'] == 'Bowl')]
ans.drop_duplicates('item_name')
#34 df의 item_name 컬럼 값이 Steak Salad 또는 Bowl 인 데이터를 데이터 프레임화 한 후, item_name를 기준으로 중복행이 있으면 제거하되 마지막 케이스만 남겨라
ans=df.loc[(df['item_name'] =='Steak Salad')|(df['item_name'] == 'Bowl')]
ans.drop_duplicates('item_name',keep='last')
#35 df의 데이터 중 new_price값이 new_price값의 평균값 이상을 가지는 데이터들을 인덱싱하라
df.loc[df.new_price>=df.new_price.mean()]

#########36 df의 데이터 중 item_name의 값이 Izze 데이터를 Fizzy Lizzy로 수정하라
df.loc[df.item_name =='Izze','item_name'] = 'Fizzy Lizzy'

###36 df의 데이터 중 choice_description 값이 NaN 인 데이터의 갯수를 구하여라
df.choice_description.isnull().sum()

#38 df의 데이터 중 choice_description 값이 NaN 인 데이터를 NoData 값으로 대체하라(loc 이용)
df.loc[df.choice_description.isnull(),'choice_description']='NoData'
df

#39 df의 데이터 중 choice_description 값에 Black이 들어가는 경우를 인덱싱하라

Ans = df[df.choice_description.str.contains('Black')]
Ans.head(5)
df[df.choice_description.str.contains('Black')].head()
#객체에 안넣고 바로 출력하면 결과가 이상하게 나온다

#40
Ans = len(df[~df.choice_description.str.contains('Vegetables')])
Ans

#41
ans = df[df.item_name.str.startswith('N')]
ans.head()

#42
ans = df[df.item_name.str.len()>=15]
ans

#43
lst =[1.69, 2.39, 3.39, 4.45, 9.25, 10.98, 11.75, 16.98]
ans=df[df.new_price.isin(lst)]
anslen = len(ans)
print(ans)
print(anslen)

#44
df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/AB_NYC_2019.csv')
df
#45
ans=df.groupby('host_name').size()
ans = df.host_name.value_counts().sort_index()
ans.head

#46
df['counts']=df.host_name.value_counts().sort_index(ascending=False)
df.head(5)
Ans = df.groupby('host_name').size().to_frame().rename(columns={0:'counts'}).sort_values('counts',ascending=False)
Ans.head()

#47
Ans = df.groupby(['neighbourhood_group','neighbourhood'], as_index=False).size()
Ans.head()

#48
Ans= df.groupby(['neighbourhood_group','neighbourhood'], as_index=False).size().groupby(['neighbourhood_group'], as_index=False).max()
Ans.head()

#49 neighbourhood_group 값에 따른 price값의 평균, 분산, 최대, 최소 값을 구하여라
Ans = df[['neighbourhood_group','reviews_per_month']].groupby('neighbourhood_group').agg(['mean','var','max','min'])
Ans.head()

#50
Ans = df[['neighbourhood_group','reviews_per_month']].groupby('neighbourhood_group').agg(['mean','var'])
Ans.head()
#51
Ans = df.groupby(['neighbourhood','neighbourhood_group']).price.mean()
Ans.head()

#52
Ans = df.groupby(['neighbourhood','neighbourhood_group']).price.mean().unstack()
Ans.head()

#53
Ans = df.groupby(['neighbourhood','neighbourhood_group']).price.mean().unstack().fillna(-999)
Ans.head()

#54
Ans=df[df.neighbourhood_group=='Queens'].groupby('neighbourhood').price.agg(['mean','var','max','min'])
Ans.head()

#######55##########################################
Ans=df[['neighbourhood_group','room_type']].groupby(['neighbourhood_group','room_type']).size().unstack()
Ans.loc[:,:] = (Ans.values /Ans.sum(axis=1).values.reshape(-1,1))
Ans

#57
import pandas as pd
df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/BankChurnersUp.csv',index_col=0)
Ans =df.shape
df.head()
dic={
    'Unknown' :'N',
    'Less than $40K' : 'a',
    '$40K - $60K' : 'b',
    '$60K - $80K' : 'c',
    '$80K - $120K' : 'd',
    '$120K +' : 'e',
}

df['newIncome']=df.Income_Category.map(lambda x:dic[x])

#58
def change(x):
    if x == 'Unknown':
        return 'N'
    elif x=='Less than $40K':
        return 'a'
    elif x=='$40K - $60K' :
        return 'b'
    elif x=='$60K - $80K' :
        return 'c'
    elif x=='$80K - $120K' :
        return 'd'
    elif x=='$120K +' :
        return 'e'
df['newIcome']=df.Income_Category.apply(change)
Ans=df['newIncome']
Ans.head()

#59

df['AgeState']=df.Customer_Age.map(lambda x:x//10*10)
Ans=df['AgeState'].value_counts().sort_index()
Ans

#60
df['newEduLevel']=df.Education_Level.map(lambda x:1 if'Graduate' in x else 0)
df['newEduLevel'].value_counts()