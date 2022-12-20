import numpy as np
import pandas as pd
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)
# 10.1 GroupBy Mechanics
'''
by Hadley Wickham
split-apply-combine 이라는 그룹연산에 대한 용어
그룹연산 단계
1. Series, DataFrame 같은 pandas 객체나 다른 객체에 들어 있는 데이터를 하나 이상의 키를
기준으로 분리
2. 객체는 하나의 축을 기준으로 분리하고 나서 함수를 각 그룹에 적용시켜 새로운 값을 얻는다.
3. 함수를 적용한 결과를 하나의 객체로 결합한다.
각 그룹의 색인은 다음의 다양한 형태가 될 수 있으며, 모두 같은 타입일 필요도 없다.
- 그룹으로 묶을 축과 동일한 길이의 리스트나 배열
- DataFrame 의 컬럼 이름을 지칭하는 값
- 그룹으로 묶을 값과 그룹 이름에 대응하는 사전이나 Series 객체
- 축 색인 혹은 색인 내의 개별 이름에 대해 실행되는 함수
'''
# DataFrame 으로 표현되는 간단한 표 형식의 데이터

df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
 'key2' : ['one', 'two', 'one', 'two', 'one'],
 'data1' : np.random.randn(5),
 'data2' : np.random.randn(5)})
df
# 이 데이터를 key1 으로 묶고 각 그룹에서 data1 의 평균을 구하기
grouped = df['data1'].groupby(df['key1'])
grouped
# 이 grouped 변수는 GroupBy 객체
# 이 객체는 그룹 연산을 위해 필요한 모든 정보를 가지고 있어서 각 그룹에 어떤 연산을 적용할 수
# 있게 해준다.
# 그룹별 평균을 구하기 위해 GroupBy 객체의 mean 메서드 사용
grouped.mean()
# 새롭게 생성된 Series 객체의 색인은 'key1'인데, 그 이유는 DataFrame 컬럼인 df['key1'] 때문
# 만약 여러 개의 배열을 리스트로 넘겼다면 조금 다른 결과를 얻었을 것
means = df['data1'].groupby([df['key1'], df['key2']]).mean()
means
# 데이터를 두개의 색인으로 묶었고, 그 결과 계층적 색인을 가지는 Series 를 얻음.
means.unstack()

# 이 예제에서 그룹의 색인 모두 Series 객체인데, 길이만 같다면 어떤 배열도 상관없다.
states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
df['data1'].groupby([states, years]).mean()
# 한 그룹으로 묶을 정보는 주로 같은 DataFrame 안에서 찾게 된다.
# 이 경우 컬럼명(문자열, 숫자 혹은 다른 파이썬 객체)을 넘겨서 그룹의 색인으로 사용할 수 있다.
df.groupby('key1').mean()
df.groupby(['key1', 'key2']).mean()
# 위에서 df.groupby('key1').mean()코드를 보면 key2 컬럼이 결과에서 빠져있다.
# 그 이유는 df['key2']는 숫자 데이터가 아니기 때문에 이런 컬럼은 nuisance column 이라고
# 부르며 결과에서 제외
# 기본적으로 모든 숫자 컬럼이 수집되지만 원하는 부분만 따로 걸러내는 것도 가능
# GroupBy 메서드는 그룹의 크기를 담고 있는 Series 를 반환하는 size 메서드
df.groupby(['key1', 'key2']).size()
# *그룹 색인에서 누락된 값은 결과에서 제외된다.

# 10.1.1 Iterating Over Groups
# GroupBy 객체는 iteration 을 지원. 그룹이름과 그 에 따른 데이터 묶음을 튜플로 반환
for name, group in df.groupby('key1'):
 print(name)
 print(group)
# 색인이 여럿 존재하는 경우 튜플의 첫 번째 원소가 색인값이 된다.
for (k1, k2), group in df.groupby(['key1', 'key2']):
 print((k1, k2))
 print(group)
# 이 안에서 원하는 데이터만 골라낼 수 있다.
# 한 줄이면 그룹별 데이터를 사전형(dict)으로 변환하여 사용 가능
pieces = dict(list(df.groupby('key1')))
pieces['b']
# groupby 메서드는 기본적으로 axis=0 에 대해 그룹을 만든다
# 다른 축으로 그룹을 만드는 것도 가능
# 예제에서 df 의 컬럼을 dtype 에 따라 그룹으로 묶을 수 있다.
df.dtypes
grouped = df.groupby(df.dtypes, axis=1)
# 그룹을 아래처럼 출력 가능
for dtype, group in grouped:
 print(dtype)
 print(group)
# 10.1.2 Selecting a Column or Subset of Columns
# DataFrame 에서 만든 GroupBy 객체를 컬럼 이름이나 컬럼 이름이 담긴 배열로 색인하면 수집을 위해
# 해당컬럼을 선택하게 된다.
df.groupby('key1')['data1']
df.groupby('key1')[['data2']]
# 아래 코드도 같은 결과 산출
df['data1'].groupby(df['key1'])
df[['data2']].groupby(df['key1'])
# 대용량 데이터를 다룰 경우 소수의 컬럼만 집계하고 싶은 경우
df.groupby(['key1', 'key2'])[['data2']].mean()

# 색인으로 얻은 객체는 groupby 메서드에 리스트나 배열을 넘겼을 경우
# DataFrameGroupBy 객체가 되고, 단일 값으로 하나의 컬럼 이름만 넘겼을 경우 SeriesGroupBy
# 객체가 된다.
s_grouped = df.groupby(['key1', 'key2'])['data2']
s_grouped
s_grouped.mean()


# 10.1.3 Grouping with Dicts and Series
# 그룹 정보는 배열이 아닌 형태로 존재하기도 한다.
# DataFrame 예제
people = pd.DataFrame(np.random.randn(5, 5),
 columns=['a', 'b', 'c', 'd', 'e'],
 index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
people.iloc[2:3, [1, 2]] = np.nan # Add a few NA values
people
# 각 컬럼을 나타낼 그룹 목록이 있고 그룹별로 컬럼의 값을 모두 더하는 경우
mapping = {'a': 'red', 'b': 'red', 'c': 'blue',
 'd': 'blue', 'e': 'red', 'f' : 'orange'}
# dict 에서 groupby 메서드로 배열을 추출할 수 있지만
# 이 dict 에 groupby 메서드를 적용
by_column = people.groupby(mapping, axis=1)
by_column.sum()
# Series 에 대해서도 같은 기능 수행 가능
map_series = pd.Series(mapping)
map_series
people.groupby(map_series, axis=1).count()

# 10.1.4 Grouping with Functions
# 그룹 색인으로 넘긴 함수는 색인값 하나마다 한 번씩 호출되며,
# 반환값은 그 그룹의 이름으로 사용
# 이전 예제에서 people DataFrame 은 사람의 이름을 색인값으로 사용.
# 만약 이름의 길이별로 그룹을 묶고 싶다면 일므의 길이가 담긴 배열을 만들어 넘기는 대신
# len()함수 사용
people.groupby(len).sum()
# 내부적으로 모두 배열로 변환되므로 함수를 배열, dict 또는 Series 와 섞어 쓰더라도 문제가 되지
# 않는다.
key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len, key_list]).min()
# 10.1.5 Grouping by Index Levels
# 계층적으로 색인된 데이터는 축 색인의 단계 중 하나를 사용해서 편리하게 집계할 수 있는 기능 제공
columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
 [1, 3, 5, 1, 3]],
 names=['cty', 'tenor'])
hier_df = pd.DataFrame(np.random.randn(4, 5), columns=columns)
hier_df
# 이 기능을 사용하려면 level 예약어를 사용해서 레벨 번호나 이름을 사용
hier_df.groupby(level='cty', axis=1).count()

df
grouped = df.groupby('key1')
grouped['data1'].quantile(0.9)

# 자신의 데이터 집계함수를 사용하려면 배열의 aggregate 나 agg 메서드에 해당 함수를 넘기면 된다.
def peak_to_peak(arr):
 return arr.max() - arr.min()
grouped.agg(peak_to_peak)
# describe 메서드는 데이터를 집계하지 않는데도 잘 작동함을 확인할 수 있다.
grouped.describe()

# 10.2.1 Column-Wise and Multiple Function Application
# 앞에서 살펴본 팁 데이터를 다시 고려하자
# 여기서 read_csv()함수로 데이터를 불러온 다음 팁의 비율을 담기 위한 컬럼인 tip_pct 를 추가
tips = pd.read_csv('pandas_dataset2/tips.csv')
tips

# Add tip percentage of total bill
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips[:6]
# Series 나 DataFrame 의 모든 컬럼을 집계하는 것은 mean 이나 std 같은 메서드를 호출하거나
# 원하는 함수에
# aggregate 를 사용하는 것이다.
# 하지만 컬럼에 따라 다른 함수를 사용해서 집계를 수행하거나 여러 개으 함수를 한 번에 적용하기
# 원한다면
# 이를 쉽게 수행할 수 있다.
# tips 를 day 와 smoke 별로 묶어보자.
grouped = tips.groupby(['day', 'smoker'])
# 표 10-1 에서의 함수 이름을 문자열로 넘기면 된다.
grouped_pct = grouped['tip_pct']
grouped_pct.agg('mean')
# 만약 함수 목록이나 함수 이름을 넘기면 함수 이름을 컬럼 이름으로 하는 DataFrame 을 얻는다
grouped_pct.agg(['mean', 'std', peak_to_peak])
# 여기서는 데이터 그룹에 대해 독립적으로 적용하기 위해 agg 에 집계함수들의 리스트를 넘겼다.
# GroupBy 객체에서 자동으로 지정하는 컬럼 이름을 그대로 쓰지 않아도 된다.
# lamda 함수는 이름이 '<lamda>'인데 이를 그대로 쓸 경우 알아보기 힘들어진다.
# 이때 이름과 함수가 담긴(name, function) 튜플의 리스트를 넘기면
# 각 튜플에서 첫 번째 원소가 DataFrame 에서 컬럼 이름으로 사용된다.
# (2 개의 튜플을 가지는 리스트가 순서대로 매핑된다.)
grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])
 # DataFrame 은 컬럼마다 다른 함수를 적용하거나 여러 개의 함수를 모든 컬럼에 적용할 수 있다.
# tip_pct 와 total_bill 컬럼에 대해 동일한 세 가지 통계를 계산한다고 가정
functions = ['count', 'mean', 'max']
result = grouped['tip_pct', 'total_bill'].agg(functions)
result
# 위에서 반환된 DataFrame 은 계층적인 컬럼을 가지고 있으며 이는 각 컬럼을 따로 계산한 다음
# concat 메서드를 이용해서 keys 인자로 컬럼 이름을 넘겨서 이어 붙인 것과 동일하다.
result['tip_pct']
# 위에서 처럼 컬럼 이름과 메서드가 담긴 튜플의 리스트를 넘기는 것도 가능
ftuples = [('Durchschnitt', 'mean'), ('Abweichung', np.var)]
grouped['tip_pct', 'total_bill'].agg(ftuples)
# 컬럼마다 다른 함수를 적용하고 싶다면 agg 메서드에 커럼 이름에 대응하는 함수가 들어있는 dict 를
# 넘기면 된다.
grouped.agg({'tip' : np.max, 'size' : 'sum'})
grouped.agg({'tip_pct' : ['min', 'max', 'mean', 'std'],
 'size' : 'sum'})
# 10.2.2 Returning Aggregated Data Without Row Indexes
# 지금까지 예제에서 집계된 데이터는 유일한 그룹키 조합으로 색인(어떤 경우에는 계층적 색인)되어
# 반환되었다.
# groupby 메서드에 as_index=False 를 넘겨서 색인되지 않도록 할 수 있다.
tips.groupby(['day', 'smoker'], as_index=True).mean()
# 색인된 결과에 대해 reset_index 메서드를 호출해서 같은 결과를 얻을 수 있다.
# as_index=False 옵션을 사용하면 불필요한 계산을 피할 수 있다.