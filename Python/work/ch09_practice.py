import urllib.request
from bs4 import BeautifulSoup

url ='https://www.naver.com/'

res = urllib.request.urlopen(url)
data = res.read()

src = data.decode("utf-8")
print(src)

html = BeautifulSoup(src, 'html.parser')
print(html)

a = html.find('a')
print('a tag :',a)
print('a tag 내용:',a.string)


#############
#태그 속성 찾기
############

import urllib.request
from bs4 import BeautifulSoup

url ='https://www.naver.com/'
res = urllib.request.urlopen(url)
data = res.read()
src = data.decode("utf-8")
html = BeautifulSoup(src, 'html.parser')

links= html.find_all('a')
print('links size=',len(links))

for link in links:
    try:
        print(link.attrs['href'])
        print(link.attrs['target'])
    except Exception as e:
        print('예외발생 :',e)

import re
print('패턴객체 이용 속성 찾기')
patt = re.compile('http://')
links = html.find(href=patt)
print (links)


###############
#news자료 수집
##############
import urllib.request
from bs4 import BeautifulSoup

url='http://media.daum.net'
res = urllib.request.urlopen(url)
source = res.read()

source = source.decode("utf-8")
html = BeautifulSoup(source,'html.parser')

atags = html.select('a[class=link_txt]')
print('a tag 수:',len(atags))

crawling_data=[]

cnt =0
for atag in atags:
    cnt+=1
