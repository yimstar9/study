#임성구
#
# 1. 다음의 벡터 EMP는 ‘입사연도이름급여’순으로 사원의 정보가 기록된 데이터이다. 벡터
# EMP를 이용하여 다음과 같은 출력 결과가 나타나도록 함수를 정의하시오.
# EMP <- c(“2014홍길동220”, “2002이순신300”, “2010유관순260”)

# stringr 패키지: str_extract(), str_replace()함수
# 숫자변한함수: as.numeric()함수
# 한글 문자 인식 정규표현식 패턴: [가-힣]


emp_pay<-function(x)
{
  library(stringr)
  pay <- numeric()
  name <- character()
  idx <- 1

  for(n in x)
  {
    name[idx] <- str_extract(n, '[가-힣]{3}')
    pay1 <- str_extract(n, '[가-힣]{3}[0-9]{3}')
    pay1 <- str_replace(pay1, '[가-힣]{3}', '')
    pay2 <- as.numeric(pay1)

    pay[idx] <- pay2
    idx <- idx + 1

  }
  avg <- mean(pay2)
  cat('전체급여 평균 :', avg, '\n')
  print('평균 이상 급여 수령자')

  n <- 1:length(x)
  for(i in n)
  {
    if(pay[i] >= avg)
    {

      cat(name[i],"==>",pay[i],'\n')
    }
  }
}

EMP <-  c("2014홍길동220", "2002이순신300", "2010유관순260")
emp_pay(EMP)


# 2. 다음 조건에 맞게 client 데이터프레임을 생성하고, 데이터를 처리하시오

# 1) 3개의 벡터 객체를 이용하여 client 데이터프레임을 생성하시오.
# 2) price변수의 값이 65만원 이상이며 문자열 “beat”, 65만원 미만이면 문자열 “Normal”
# 을 변수 result에 추가하시오. (힌트, ifelse()사용)
# 3) result변수를 대상으로 빈도수를 구하시오

name <- c("유관순", "홍길동", "이순신", "신사임당")
gender <- c("F", "M", "M", "F")
price <- c(50, 65, 45, 75)

client <- data.frame(name,gender,price);client
client$result <- ifelse(client$price >= 65, "Beat", "Normal");client
table(client$result)



