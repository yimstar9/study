import pandas as pd
df = pd.read_csv('ex1.csv')
df
# 구분자를 쉼표로 지정
pd.read_table('ex1.csv', sep=',')
# type examples/ex2.csv
pd.read_csv('ex2.csv', header=None)
# 컬럼명 지정
pd.read_csv('ex2.csv', names=['a', 'b', 'c', 'd', 'message'])
# message 컬럼을 색인으로 하는 DAtaFrame 을 반환하려면 index_col 인자에 4 번째 컬럼
# 또는 'message'이름을 가진 컬럼을 지정하여 색인으로 만듦
names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv('ex2.csv', names=names, index_col='message')
pd.read_csv('pandas_dataset2/ex4.csv')

sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
pd.read_csv('pandas_dataset2/ex5.csv', na_values=sentinels)
pd.read_csv('pandas_dataset2/ex5.csv')
result = pd.read_csv('pandas_dataset2/ex5.csv', na_values=['world'])
result

pd.options.display.max_rows = 65
pd.options.display.min_rows =20
result = pd.read_csv('pandas_dataset2/ex6.csv')
result

# 예로 ex6.csv 파일을 순회하면서 'key'로우에 있는 값을 세어보려면 다음과 같이 한다.
chunker = pd.read_csv('pandas_dataset2/ex6.csv', chunksize=1000)
tot = pd.Series([])
for piece in chunker:
 tot = tot.add(piece['key'].value_counts(), fill_value=0)
tot = tot.sort_values(ascending=False)
tot[:10]

# 컬럼의 일부분만 기록하거나 순서 지정
import sys
import numpy as np
data = pd.read_csv('pandas_dataset2/ex5.csv')

data.to_csv(sys.stdout, sep='|')
# 결과에서 누락된 값은 비어 있는 문자열로 나타나는데 원하는 값으로 지정 가능
data.to_csv(sys.stdout, na_rep='NULL')
# 다른 옵션을 명시하지 않으면 로우와 컬럼 이름이 기록된다. 로우와 컬럼 이름을 포함하지 않을 경우

data.to_csv(sys.stdout, index=False, header=False)
# 컬럼의 일부분만 기록하거나 순서 지정
data
data.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])
dates = pd.date_range('1/1/2000', periods=7)
ts = pd.Series(np.arange(7), index=dates)
ts.to_csv('tseries.csv')

