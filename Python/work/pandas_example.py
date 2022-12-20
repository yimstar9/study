import pandas as pd
import numpy as np
# 10.3.5 Example: Group Weighted Average and Correlation
df = pd.DataFrame({'category': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'], 'data': np.random.randn(8), 'weights': np.random.rand(8)})
df
grouped = df.groupby('category')

get_wavg = lambda g: np.average(g['data'], weights=g['weights'])
grouped.apply(get_wavg)
close_px = pd.read_csv('pandas_dataset2/stock_px_2.csv', parse_dates=True,
 index_col=0)
close_px.info()
close_px[-4:]
spx_corr = lambda x: x.corrwith(x['SPX'])
rets = close_px.pct_change().dropna()
get_year = lambda x: x.year
by_year = rets.groupby(get_year)
by_year.apply(spx_corr)
by_year.apply(lambda g: g['AAPL'].corr(g['MSFT']))


# 10.3.6 Example: Group-Wise Linear Regression
# install 'statsmodels' 라이브러리
import statsmodels.api as sm
def regress(data, yvar, xvars):
 Y = data[yvar]
 X = data[xvars]
 X['intercept'] = 1.
 result = sm.OLS(Y, X).fit()
 return result.params
by_year.apply(regress, 'AAPL', ['SPX'])