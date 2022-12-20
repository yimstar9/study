from re import findall
from statistics import mean
import numpy as np
emp = ["2014홍길동220", "2002이순신300", "2010유관순260", ]
names=[]
pays=[]
def pay_pro(emp):
    dic = {}
    arr=None
    for e in emp:
        re = findall('[가-힣]{3}[0-9]{3}', e)
        name = findall('[가-힣]{3}', re[0])
        names.extend(name)
        pay=findall('[0-9]{3}', re[0])
        pays.extend(pay)
        # dic[name[0]]=(int(pay[0]))
    result = mean(pays)

    print(arr)
    # print(dic)
    print('전체 사원 급여 평균 :', result)
    names[pays>=result]
    # for j in dic:
    #     if dic[j] >= result:
    #         print(j, "=>", dic[j])
    return

pay_pro(emp)