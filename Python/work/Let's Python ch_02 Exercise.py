# 01
su=5
dan=800
sum = su * dan
print("su 주소 : %s" %(id(su)))
print("dan 주소 :",format(id(dan)))
print(f"금액 : {sum}")

# 02
X=2
Y=2.5*X**2+3.3*X+6
Y

# 03
fat=int(input("지방의 그램을 입력하세요 : "))
car=int(input("탄수화물의 그램을 입력하세요 : "))
pro=int(input("단백질의 그램을 입력하세요 : "))
cal = fat*9+car*4+pro*4
#print("총칼로리 : ",format(cal,",d"), end=' cal')
print(f"총칼로리 : {cal:,d}", end=' cal')

#04
word1 = input("첫번째 단어 : ")
word2 = input("두번째 단어 : ")
word3 = input("세번째 단어 : ")
print("="*20)
abbr = word1[0]+word2[0]+word3[0]
print(f"약자 : {abbr}")

#04-2
word4 = "Korea"
word5 = "Baseball"
word6 = "Orag"
word7 = "Victory"
print("="*20)
ans = word5[-1].upper()+word6[0].lower()+word7[0].lower()+word4[-2]
print(ans)