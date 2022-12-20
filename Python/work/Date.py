import datetime
import pandas as pd

NextMonthBegin=pd.datetime.now().date()+pd.offsets.MonthBegin()
NextMonthEnd=pd.datetime.now().date()+pd.offsets.MonthBegin()+pd.offsets.MonthEnd()
start=NextMonthBegin.strftime('%Y%m%d')
end=NextMonthEnd.strftime('%Y%m%d')

dt_index = pd.date_range(start=start, end=end,freq='W-THU')
next_rsrvDt_arr = dt_index.strftime("%Y%m%d").tolist()
next_rsrvDt_arr_dash = dt_index.strftime("%Y-%m-%d").tolist()
print(next_rsrvDt_arr,next_rsrvDt_arr_dash)