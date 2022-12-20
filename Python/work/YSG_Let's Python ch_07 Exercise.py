#임성구
##1번
import re

email = """hong@12.com
you2@naver.com
12kang@hanmail.net
kimjs@gmail.com
yimsg@korea.co.kr"""

from re import findall, match, compile
emaillist=[]
for e in email.split(sep='\n'):
    emaillist.append(e)
# pat = compile("^[a-z][0-9a-zA-Z]{3,}@[a-z][0-9a-zA-Z]{2,}[.][a-z]{2,3}([.][a-z]{2,3})?")
# pat = compile("[a-z][0-9a-zA-Z]{3,}@[a-z][0-9a-zA-Z]{2,}[.][a-z]{,3}[a-z.]{,3}$")
pat = compile(r"""
#아이디@호스트이름.최상위도메인.최상위도메인
   [a-z]            #아이디 첫문자 영문소문자
   [0-9a-zA-Z]{3,}  #아이디 두번째문자 영문,숫자 단어3자 이상
   [@]
   [a-z]            #호스트이름 첫문자 영문소문자
   [0-9a-zA-Z]{2,}  #호스트이름 두번째문자 영문,숫자 단어2자 이상
   [.]
   [a-z]{,3}         #최상위도메인 영문 소문자 3자리 이하
   [a-z.]{,3}$       #최상위도메인 영문 소문자 3자리 이하
""",re.VERBOSE)
sel_email=[email for email in emaillist if match(pat,email)]
print(sel_email)


#2번

from re import findall,sub
emp = ["2014홍길동220","2002이순신300","2010유관순260",]
def name_pro(v):
    emp_re = sub('[0-9]','',v)
    return emp_re

names= [name_pro(text) for text in emp]
print('names=',names)


#3번
from re import findall
from statistics import mean
emp = ["2014홍길동220", "2002이순신300", "2010유관순260", ]
def pay_pro(emp):
    names=[]
    pays=[]
    sum=0
    for e in emp:
        re=findall('[가-힣]{3}[0-9]{3}',e)
        name=findall('[가-힣]{3}',re[0])
        pay=findall('[0-9]{3}', re[0])
        names.append(name[0])
        pays.append(int(pay[0]))

    result = mean(pays)
    return result

m = pay_pro(emp)
print('전체 사원 급여 평균 :', m)

#4번
from re import findall
from statistics import mean
emp = ["2014홍길동220", "2002이순신300", "2010유관순260", ]

def pay_pro(emp):
    dic={}
    for e in emp:
        re=findall('[가-힣]{3}[0-9]{3}',e)
        name=findall('[가-힣]{3}',re[0])

        pay=findall('[0-9]{3}', re[0])

        dic[name[0]]=(int(pay[0]))
    result = mean(dic.values())
    # print(dic)
    # print(pays)
    # print(names)
    print('전체 사원 급여 평균 :', result)
    name[pay==1]
    for j in dic:
        if dic[j]>=result:
            print(j,"=>",dic[j])
    return
pay_pro(emp)

#5번
from re import findall, sub
texts=['AFAB54747,asabag?','abTTa $$;a12:2424','uysfsafA,A123&***$?']

def clean_text(text):
    text1=text.lower()                  #소문자화
    text2=sub('[,?$;:&*]+','',text1)    #특수 문자 지움
    text3=sub(' ','',text2)             #공백 지움
    text4=sub('[0-9]','',text3)         #숫자 지움
    return text4

text_result=[clean_text(text) for text in texts]
print(text_result)