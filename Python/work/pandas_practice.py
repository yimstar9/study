# 환경설정
import pandas as pd
import numpy as np
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.set_printoptions(precision=4, suppress=True)
pd.options.display.max_rows = PREVIOUS_MAX_ROWS

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
states = ['California', 'Ohio', 'Oregon', 'Texas','Texas']
obj4 = pd.Series(sdata, index=states)
obj4

# DataFrame 객체 생성
# 같은 길이의 리스트에 담긴 사전을 이용하거나 NumPy 배열을 이용
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
 'year': [2000, 2001, 2002, 2001, 2002, 2003],
 'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
# 만들어진 DataFrame 의 색인은 Series 와 같은 방식으로 자동으로 대입되며 컬럼은 정렬되어 저장된다.
frame
# head 메서드를 이용하여 처음 5 개의 로우만 출력 가능
frame.head()
# 원하는 순서대로 columns 를 지정하면 원하는 순서를 가진 DataFrame 객체 생성
pd.DataFrame(data, columns=['year', 'state', 'pop'])
# Series 와 동일하게 사전에 없는 값을 넘기면 결측치로 저장된다.
frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
 index=['one', 'two', 'three', 'four',
 'five', 'six'])
frame2
frame2.columns
# DataFrame 의 컬럼은 Series 처럼 사전 형식의 표기법으로 접근하거나 속성 형식으로 접근할 수 있다.
frame2['state']
frame2.year
frame2['eastern'] = frame2.state == 'Ohio'
frame2
frame2['size']=1
frame2.size
frame2['size']
frame2.columns
# * frame2[column] 형태로 사용하는 것은 어떤 컬럼이든 가능.
# 하지만 frame2.column 형태로 사용하는 것은 사용가능한 변수이름 형식일때만 작동
# 반환된 Series 객체가 DataFrame 과 같은 색인을 가지면 알맞은 값으로 name 속성이 채워진다.
# 로우는 위치나 loc 속성을 이용하여 이름을 통해 접근할 수 있다.
frame2.loc['three'] ##인덱스 접근
frame2.three

# Series 를 대입하면 DataFrame 의 색인에 따라 값이 대입되며 존재하지 않는 색인에는 결측치가 대입된다.
val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2
# 중첩된 사전을 이용하여 데이터 생성 가능
# 중첩된 사전
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
# 중첩된 사전을 DataFrame 에 넘기면 바깥에 있는 사전의 키는 컬럼이 되고 안에 있는 키는 로우가 된다.
frame3 = pd.DataFrame(pop)
frame3
# 데이터의 전치(transpose) 가능
frame3.T
# 중첩된 사전을 이용하여 DataFrame 을 생성할 때 안쪽에 있는 사전값은 키값별로 조합되어 결과의 색인이 되지만
# 색인을 직접 지정하면 지정된 색인으로 DataFrame 을 생성
pd.DataFrame(pop, index=[2001, 2002, 2003])

# reindex: 새로운 색인에 맞도록 객체를 새로 생성
# 예제
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj
# Series 객체에 대해 reindex 를 호출하면 데이터를 새로운 색인에 맞게 재배열하고,
# 존재하지 않는 색인 값이 있다면 NaN 을 추가한다.
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2
# 시계열 데이터를 재색인할 때 값을 보간하거나 채워 넣어야 할 경우
# method 옵션을 이용하여 실행
# ffill 메서드를 이용하여 누락된 값을 직전의 값으로 채워 넣을 수 있다.
obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3
obj3.reindex(range(6), method='ffill')
# DataFrame 에 대한 reindex 는 로우(새긴), 컬럼 또는 둘다 변경 가능
# 순서만 전달하면 로우가 재색인된다.
frame = pd.DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
frame
frame = frame.reindex(['a', 'b', 'c', 'd'])
frame
# 컬럼은 columns 예약어를 사용하여 재색인
states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)
# 재색인은 loc 를 이용하여 라벨로 색인하면 좀 더 간결하게 할 수 있다.
states = ['Ohio', 'Texas',  'California']
frame.loc[['a', 'b', 'c', 'd'], states]

'''
색인 배열, 또는 삭제하려는 로우나 컬럼이 제외된 리스트를 가지고 있다면
로우나 컬럼을 쉽게 삭제 가능.
이 방법은 데이터의 모양을 변경하는 작업이 필요
'''
# drop 메서드를 사용하면 선택한 값들이 삭제된 새로운 객체를 얻을 수 있다.
obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
obj
new_obj = obj.drop('c')
new_obj
obj.drop(['d', 'c'])
# DataFrame 에서는 로우와 컬럼 모두에서 값을 삭제할 수 있다.
# 예제
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
 index=['Ohio', 'Colorado', 'Utah', 'New York'],
 columns=['one', 'two', 'three', 'four'])
data
# drop 함수에 인자로 로우 이름을 지정하면 해당 로우 (axis())의 값을 모두 삭제
data.drop(['Colorado', 'Ohio'])
# 컬럼 값을 삭제할 때는 axis=1 또는 axis='columns'를 인자로 넘겨주면 된다.
data.drop('two', axis=1)

data.drop(['two', 'four'], axis='columns')
# drop()함수 처럼 Series 나 DataFrame 의 크기 또는 형태를 변경하는 함수는
# 새로운 객체를 반환하는 대신 원본 객체를 변경한다.
obj.drop('c', inplace=True)
obj
# inplace 옵션을 사용하는 경우 버려지는 값을 모두 삭제하므로 주의!

# Series 의 색인(obj[...])은 NumPy 배열의 색인과 유사하게 동작하지만 정수가 아니어도 된다는 점이 다르다.
# 예제
obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj
obj['b']
obj[1]
obj[2:4]
obj[['b', 'a', 'd']]
obj[[1, 3]]
obj[obj < 2]

# 라벨 이름으로 슬라이싱하면 시작점과 끝점을 포함한다는 것이 일반 파이썬에서의 슬라이싱과 다른점.

obj['b':'c']
# 슬라이싱 문법으로 선택된 영역에 값을 대입하는 것은 생각하는대로 동작
obj['b':'c'] = 5
obj


# 색인으로 DataFrame 에서 하나 이상의 컬럼 값을 가져올 수 있음.
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
 index=['Ohio', 'Colorado', 'Utah', 'New York'],
 columns=['one', 'two', 'three', 'four'])
data
data['two']
data[['three', 'one']]
# 슬라이싱으로 로우를 선택하거나 불리언 배열로 로우를 선택 가능
data[:2]
data[data['three'] > 5]
# 다른 방법으로 스칼라 비교를 이용하여 생성된 불리언 DataFrame 을 사용하여 값을 선택
data < 5
data[data < 5] = 0
data

# Selection with loc and iloc
# loc & iloc: DataFrame 의 로우에 대해 라벨로 색인하는 방법으로 특수 색인 필드
# 축의 라벨을 사용하여 DataFrame 의 로우와 컬럼을 선택 가능.
# 축 이름을 선택할 때는 loc 를, 정수색인으로 선택할 때는 iloc 사용
data.loc['Colorado', ['two', 'three']]
data.iloc[2, [3, 0, 1]]
data.iloc[2]
data.iloc[[1, 2], [3, 0, 1]]
# loc & iloc 함수는 슬라이스 지원 및 단일 라벨이나 라벨 리스트 지원
data.loc[:'Utah', 'two']
data.iloc[:, :3][data.three > 5]
data.iloc[:,:2]

# DataFrame 의 경우 정렬은 로우와 컬럼 모두에 적용됨.
df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'), index=['Ohio', 'Texas', 'Colorado'])
df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1
df2
df1 + df2
# 공통되는 컬럼 라벨이나 로우 라벨이 없는 DataFrame 을 더하면 결과에 아무것도 안 나타남.
df1 = pd.DataFrame({'A': [1, 2]})
df2 = pd.DataFrame({'B': [3, 4]})
df1
df2
df1 - df2
# Arithmetic methods with fill values
# 서로 다른 색인을 가지는 객체 간의 산술연산에서 존재하지 않는 축의 값을 특수한 값(예, 0)으로 지정 시
df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
df2.loc[1, 'b'] = np.nan
df1
df2
# 겹치지 않는 부분은 NA 값
df1 + df2
# df1 에 add 메서드 사용
df1.add(df2, fill_value=0)
# 메서드는 r 로 시작하는 짝꿍메서드를 가진다.
1 / df1
df1.rdiv(1)
# 재색인
df1.reindex(columns=df2.columns, fill_value=0)

# pandas 객체에 Numpy 의 유니버설 함수(배열의 각 원소에 적용되는 메서드) 적용 가능
frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),
 index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame
np.abs(frame)
# 자주 사용되는 또 다른 연산은 각 컬럼이나 로우의 1 차원 배열에 함수를 적용하는 것
# DataFrame 의 apply 메서드 사용
f = lambda x: x.max() - x.min() # 최대값 - 최소값
frame.apply(f, axis=0)
# apply 함수에 axis = 'columns' 설정시 각 로우에 대해 한 번씩만 수행
frame.apply(f, axis='columns')

# 배열에 대한 일반적인 통계(예, sum, mean)는 DataFrame 의 메서드로 존재하므로 apply 메서드
# 사용 불필요
def f(x):
 return pd.Series([x.min(), x.max()], index=['min', 'max'])
frame.apply(f)
# 배열의 각 원소에 적용되는 파이썬의 함수 사용 가능
# frame 객체에서 실숫값을 문자열 포맷으로 변환하고 싶다면 applymap 메서드 사용
format = lambda x: '%.2f'
frame.applymap(format)

obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
obj
obj.rank()
obj.rank(method='dense')

# DataFrame 에서 하나 이상의 컬럼에 있는 값으로 정렬을 하는 경우
# sort_value 함수의 by 옵션에 하나 이상의 컬럼 이름을 넘기면 된다.
frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame
frame.sort_values(by='b')

# pandas 의 많은 함수(예, reindex)에서 색인값은 유일해야 하지만 의무적이지 않다.
# 중복된 색인값을 가지는 Series 객체 예제
obj = pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj
# 색인이 유일한 값인지 확인
obj.index.is_unique
obj['a']
obj['c']

df = pd.DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
df
df[1]
df.loc['b']

# 예시
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]], index=['a', 'b', 'c', 'd'], columns=['one', 'two'])
df
df.sum()
df.sum(axis='columns')
# 전체 로우나 컬럼의 값이 NA 가 아니라면 NA 값을 제외하고 계산
# skipna 옵션으로 조정 가능
df.mean(axis='columns', skipna=False)
# 최소값 또는 최대값을 가지는 색인 반환
df.idxmax()
df.idxmin()
# 누적합
df.cumsum()
# 한번에 여러 개의 통계 결과 반환
df.describe()
obj = pd.Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()

DataUrl = 'https://raw.githubusercontent.com/Datamanim/pandas/main/lol.csv'
df1 = pd.read_table(DataUrl)
df1
type(df1)
df1.head(5)
df1.shape
df1.iloc[:,5].dtype
df1.index
df1.iloc[2,5]
df1['gameId']
DataUrl = 'https://raw.githubusercontent.com/Datamanim/pandas/main/Jeju.csv'
df2 = pd.read_csv(DataUrl,encoding='euc-kr')
df2
df2.tail(3)
df2.select_dtypes(exclude=object).columns
df2.isnull().sum()
df2.info()
df2.describe()
Ans  = df2['평균 속도'].quantile(0.75) -df2['평균 속도'].quantile(0.25)
Ans
len(df2['읍면동명'].unique())

DataUrl = 'https://raw.githubusercontent.com/Datamanim/pandas/main/chipo.csv'
df3 = pd.read_csv(DataUrl)
type(df3)
df3.columns

df3.loc[df3['quantity']==3]

price = pd.read_pickle('yahoo_price.pkl')
volume = pd.read_pickle('yahoo_volume.pkl')

import pandas_datareader.data as web
all_data = {ticker: web.get_data_yahoo(ticker)
 for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']}
price = pd.DataFrame({ticker: data['Adj Close'] for ticker, data in
all_data.items()})
volume = pd.DataFrame({ticker: data['Volume'] for ticker, data in
all_data.items()})

returns = price.pct_change()
returns.tail()
returns['MSFT'].corr(returns['IBM'])
# 공분산 계산
returns['MSFT'].cov(returns['IBM'])
# 파이썬 속성 이름 규칙에 어긋나지 않아 좀 더 편리한 문법으로 해당 컬럼 선택 가능
returns.MSFT.corr(returns.IBM)
# DataFrame 에서 corr 과 cov 메서드는 DataFrame 행렬에서 상관관계와 공분산을 계산
returns.corr()
# DataFrame 에서 corrwith 메서드 사용 시 다른 series 나 DataFrame 과의 상관관계를 계산
# Series 를 넘기면 각 컬럼에 대해 계산한 상관관계를 담고 있는 Series 반환
returns.corrwith(returns.IBM)
# DataFrame 을 넘기면 맞아떨어지는 컬럼 이름에 대한 상관관계 계산
returns.corrwith(volume)

# 예시
obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
uniques = obj.unique()
uniques

# 유일값은 정렬된 순서대로 반홙되지 않지만 필요하다면 uniques.sort()를 이용하여 정렬
uniques.sort()
uniques

obj.value_counts()
# 내림차순으로 정렬
# value_counts 메서드는 pandas 의 최상위 메서드로 어떤 배열이나 순차 자료구조에서도 사용 가능
pd.value_counts(obj.values, sort=False)
# isin 메서드는 어떤 값이 Series 에 존재하는지 나타내는 불리언 벡터 반환
# Series 나 DataFrame 의 컬럼에서 값을 골라내고 싶을 때 사용
obj
mask = obj.isin(['b', 'c'])
mask
obj[mask]
import csv
f = open('pandas_dataset2/ex7.csv')
reader = csv.reader(f)
# 큰 따옴표가 제거된 튜플 얻을 수 있다.
for line in reader:
 print(line)
# 원하는 형태로 데이터를 넣을 수 있도록 하자.
# 파일을 읽어 줄 단위 리스트로 저장
with open('pandas_dataset2/ex7.csv') as f:
 lines = list(csv.reader(f))
# 헤더와 데이터 구분
header, values = lines[0], lines[1:]
# 사전표기법과 로우를 컬럼으로 전치해주는 zip(*values)이용 데이터 컬럼 사전 만들기
data_dict = {h: v for h, v in zip(header, zip(*values))}
data_dict
# csv 파일은 다양한 형태로 존재할 수 있다. 다양한 구분자, 문자열을 둘러싸는 방법, 개행 문자 같은
# 것들은
# csv.Dialect 를 상속받아 새로운 클래스를 정의해서 해결
class my_dialect(csv.Dialect):
 lineterminator = '\n'
 delimiter = ';'
 quotechar = '"'
 quoting = csv.QUOTE_MINIMAL
# reader = csv.reader(f, dialect=my_dialect)
reader = csv.reader('ex7.csv', dialect=my_dialect)
# 서브클래스를 정의하지 않고 csv.readr 에 키워드 인자로 각각의 csv 파일의 특징을 지정해서
# 전달해도 된다.
# reader = csv.reader(f, delimiter='|')
reader = csv.reader('ex7.csv', delimiter='|')

# 히스토그램
data = pd.DataFrame({'Qu1': [1, 3, 4, 3, 4], 'Qu2': [2, 3, 1, 2, 3], 'Qu3': [1, 5, 2, 4, 4]})
data
data.value_counts()

result = data.apply(pd.value_counts, axis='columns')
result = data.apply(pd.value_counts).fillna(0)
result
##apply 함수가 for문이랑 비슷

# 예제
data = pd.DataFrame(np.arange(12).reshape((3, 4)),
 index=['Ohio', 'Colorado', 'New York'],
 columns=['one', 'two', 'three', 'four'])
data
# 축 색인에도 map 메서드 사용
transform = lambda x: x[:4].upper()
data.index.map(transform)
# 대문자로 변경된 축 이름을 DataFrame 의 index 에 바로 대입
data.index = data.index.map(transform)
data
# 원래 객체를 변경하지 않고 새로운 객체 생성 시 rename 메서드 사용
data.rename(index=str.title, columns=str.upper)
# dic 객체를 이용하여 축 이름 중 일부만 변경 가능
data.rename(index={'OHIO': 'INDIANA'},
 columns={'three': 'peekaboo'})
data # 원본 유지
# rename 메서드를 사용하면 DataFrame 을 직접 복사해서 index 와 columns 속성을 갱신할 필요 없이
# 바로 변경 가능
# 원본 데이터를 바로 변경하려면 inplace=True 옵션 사용
data.rename(index={'OHIO': 'INDIANA'}, inplace=True)
data # 원본 수정