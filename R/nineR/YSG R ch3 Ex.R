###임성구

# 1. R에서 제공하는 CO2 데이터 셋을 대상으로 다음과 같은 단계로 파일에 저장하시오.
# 1단계: Treatment 컬럼 값이 ‘nonchilled’ 인 경우 ‘CO2_df1.csv’ 파일로 행번호를 제외하
# 고 저장한다.
# 2단계: Treatment 컬럼 값이 ‘chilled’인 경우 ‘CO2_df2.csv’파일로 행 번호를 제외하고 저
# 장한다.
data(CO2)
non <- subset(CO2,Treatment=='nonchilled')
write.csv(non, "CO2_df1.csv", row.names = F)
chilled <- subset(CO2,Treatment=='chilled')
write.csv(chilled, "CO2_df2.csv", row.names = F)


# 2. 본문에서 작성한 titanic변수를 이용하여 다음을 실행하시오
# 1) ‘titanic.csv’파일을 titanicData변수로 가져와서 결과를 확인하고, titanicData의 관측치와
# 컬럼수를 확인힌다. (힌트, str()함수 이용)
# 2) 1, 3번 컬럼을 제외한 나머지 컬럼을 대상으로 상위 6개의 관측치를 확인한다
titanicData <-
  read.csv("https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/titanic.csv")
str(titanicData)
titanicData
head(titanicData[, -c(1,3)])



