a=[1,2,[3,4]]
print(a[2])
print(a[-1])


a="hello world"
len(a)
print(a[3]) #네번째 문자
print(a[-1]) #마지막 문자
print(a[0:3]) #0번째부터 2번째까지 0<=a<3
print(a[4:]) #4번째부터 끝까지

temp=33
b="오늘 오전 온도는 %d도,오후 온도는 %d도,습도는 %d%% 입니다.\n" %(26,temp,80)
c="오늘 날씨는 %s" % "덥습니다."
print(b,c)

d="%-10s파이썬\n" % "hi" #전체길이가 10칸 나머지공백 좌측정렬
e="%10s\n" % "hi" #전체길이가 10칸 나머지 공백넣고 우측정렬
f="%10.3f\n" %3.141592 #전체길이가 10칸 소수점 3자리 우측정렬
g="%.4f" %3.141592
print(d,e,f,g)

num=2
name="신"
a="나는 라면 {0}개를 계란은 {1}개 먹었다.\n".format(3,num)
b="나는 {}라면 {}개를 먹었다.\n".format(name,num)
c="나는 {a}라면 {b}개를 {c}명이서 먹었다.\n".format(a="삼양",b=4,c="네")
#{name}형식 포매팅할때는 반드시 포맷함수안에 name=value같은 형태로 입력해야 한다.
e="나는 {d}라면 {0}개를 {1}명이서 먹었다.\n".format(2,3,d="진")
#인덱스와 이름 혼용해서 사용할땐 format 함수안에 name=value형태는 제일 뒤에 와야 한다.
print(a,b,c,e)

d = {'name':'홍길동', 'age':30}
e=f'나의 이름은 {d["name"]}입니다. 나이는 {d["age"]}입니다.'
print(e)

age = 30
f= f'나는 내년이면 {age+1}살이 된다.'
print(f)


r=f'{g:>10}'
l=f'{g:<10}'

a=f'{g*5:@^9}'
print(r)
print(l)
print(c)
print(a)

g="*"
i=0
for i in range(1,6):
    print(f'{g * (2*i-1):^9}')

a=" aa bd fcccb draa  "
a.count("a")
a.find("z")
a.index('r')
",".join(a)
a.split(' ')
a.upper()
a.strip()

a = [1, 2, 3, ['a', 'b', 'c']]
a[0]
a[-1]
a[3][1]
a[2:4]
a*2
len(a)
str(a[2])+"hi" #a[2]저장된 값은 정수형이라서 hi문자열과 덧셈을 할 수 없다 str로 문자열로 변환후 덧셈
a[1]=23
a
del a[1]
a.append([4,5])
b=[5,1,6,7,2,9]
b.sort() #오름차순
b.reverse() #역순정렬
b.insert(2,11) #2번 인덱스에 11추가
b.remove(5) #첫번째 5를 제거
b.pop() #리스트 마지막 요소 내보내고 지우기
b.sort(reverse=True) ##내림차순
b