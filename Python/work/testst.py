import pandas as pd
import os

print(os.getcwd())

score = pd.read_csv('data\data.csv')
print(score.info())
print(score.head())

dur = score.Duration
pul = score['Pulse']
max = score['Maxpulse']
cal = score['Calories']

print('max dur =', max(dur))
print('max pul =', max(pul))
print('max max =', max(max))
print('max cal =', max(cal))
print(cal)
from statistics import mean
print(mean(dur))