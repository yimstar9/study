#임성구
#1번
a=3
b=3
print(id(a)==id(b))

#2번 (1, 2, 3)의 튜플에 4를 추가하여 (1, 2, 4, 3)처럼 만들어 출력하고자 합니다. 밑줄을 채우시오.(한줄로 표현하시오)
a=(1,2,3)
a=a[:2]+(4,3)
print(a)

#3번 f문자열 포매팅을 이용하여 아래 출력결과대로 출력되도록 문자열을 처리하시오.
print(f'{" python3! ":=^14}')

#4번 "홍길동 씨의 주민등록번호는 000101-1023456이다."
pin="000101-1023456"
mon=pin[2:4]
date=pin[4:6]
print(f"홍길동씨의 생일은 {mon}월{date}일이다")

#5번 본인 영문이름 출력

Chr1 = 'ABCDEFG'
Chr2 = 'HIJKLMN'
Chr3 = 'OPQRST'
Chr4 = 'UVWXYZ'

a= Chr4[4]+Chr2[1].lower()+Chr2[-2].lower()+Chr3[-2]+Chr4[0].lower()+Chr2[-1].lower()+Chr1[-1].lower()+Chr1[-1]+Chr3[0].lower()+Chr3[0].lower()
print(a)

