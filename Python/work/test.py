import datetime
import pandas as pd

NextMonthBegin=pd.datetime.now().date()+pd.offsets.MonthBegin()
NextMonthEnd=pd.datetime.now().date()+pd.offsets.MonthBegin()+pd.offsets.MonthEnd()
start=NextMonthBegin.strftime('%Y%m%d')
end=NextMonthEnd.strftime('%Y%m%d')

dt_index = pd.date_range(start=start, end=end,freq='W-THU')
next_rsrvDt_arr = dt_index.strftime("%Y%m%d").tolist()
next_rsrvDt_arr_dash = dt_index.strftime("%Y-%m-%d").tolist()
next_rsrvDt_arr
next_rsrvDt_arr_dash

def a():
    for i in range(len(next_rsrvDt_arr)):
        status_code = '302'
        while '302' in status_code:
            status_code = 계정1.selectRentInfo(i, session)

            if '302' not in status_code:
                print('예약 성공:' + next_rsrvDt_arr[i], datetime.datetime.now())
                break
    return


from statistics import mean
from math import sqrt
x=[5,9,1,7,4,6]
class Scattering:
    def __init__(self,x):

    def var_func(self):

        return
    def std_func(self):

        return

cal = Scattering(x)
print("분산:",cal.var_func())
print("표준편차:",cal.std_func())


a=[1,2,3,4]
b=list(range(len(a)))
b
range(len(a))
list(range(int(len(a))))
list(range(10))
list(10)