import re
from re import findall
st1 = '1234 abc홍길동 ABC_555_6 이사도시'
print(findall('1234',st1))
print(findall('[0-9]',st1))
print(findall('[0-9]{3}',st1))
print(findall('[0-9]{3,}',st1))
print(findall('\\d{3,}',st1))

print(findall('[가-힣]',st1))
print(findall('[가-힣]{3,}',st1))
print(findall('[a-z]{3}',st1))
print(findall('[a-z|A-z]{3}',st1))

str2 = 'test1abcABC 123mbc 45test'
print(findall('^test',str2))  #test로 시작하는 문자열인데 왜 test1abcABC가 안나오지?

from re import split, match, compile
multi_line = """http://www.naver.com
http://www.daum.net
google.com"""
website=split('\n',multi_line)
p=compile(".+:*")
m=p.search(website[0] )
print(m.group())
#########
from re import split, match, compile
p = compile(r"(\w+)\s+((\d+)[-]\d+[-]\d+)")
m = p.search("park 010-1234-1234")
print(m.group())

###############################
s = '<html><head><title>Title</title>'
len(s)

print(re.match('<.*>', s).span())

print(re.match('<.*>', s).group())
print(re.match('<.+>?', s).group())
# *?, +?, ??
#'*', '+' 및 '?' 한정자는 모두 탐욕적 (greedy)입니다;
# 가능한 한 많은 텍스트와 일치합니다. 때로는 이 동작이 바람직하지 않습니다;
# RE <.*>를 '<a> b <c>'와 일치시키면, '<a>'가 아닌 전체 문자열과 일치합니다.
# 한정자 뒤에 ?를 추가하면 비 탐욕적 (non-greedy) 또는 최소 (minimal) 방식으로 일치를 수행합니다;
# 가능하면 적은 문자가 일치합니다. RE <.*?>를 사용하면 '<a>' 만 일치합니다.

from re import match, search, compile
data =["이유덕","이재영","권종표","이재영","박민호",'강상희','이재영','김지완','최승혁','이성연','박영서','박민호','전경헌','송정환','김재성','이유덕','전경헌']
#김씨와 이씨는 각각 몇 명 인가요?
kim=[]
lee=[]
k = compile('[김]..')
l = compile('[이]..')
for i in data:
    if match(k,i):
        kim.append(i)
for i in data:
    if match(l,i):
        lee.append(i)
print(kim.count("김"))
print(kim,'총',len(kim),'명')
print(lee,'총',len(lee),'명')
#"이재영"이란 이름이 몇 번 반복되나요?
cnt=0
for key in lee:
    if key =='이재영': cnt+=1
print(cnt)
#중복을 제거한 이름을 출력하세요.
kimset={}
leeset={}
kimset=set(kim)
leeset=set(lee)
print(kimset)
print(leeset)
#중복을 제거한 이름을 오름차순으로 정렬하여 출력하세요.
sortedkim=[]
sortedkim=sorted(list(kimset))
sortedlee=[]
sortedlee=sorted(list(leeset))
print(sortedlee)
print(sortedkim)