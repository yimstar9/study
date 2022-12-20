#박근혜 전 대통령의 대선 출마 선언문이 들어있는 speech_park.txt를 이용해 문제를 해결해 보세요.
# Q1. speech_park.txt를 불러와 분석에 적합하게 전처리한 다음 연설문에서 명사를 추출하세요.
# Q2. 가장 자주 사용된 단어 20개를 추출하세요.
# Q3. 가장 자주 사용된 단어 20개의 빈도를 나타낸 막대 그래프를 만드세요.
# Q4. 전처리하지 않은 연설문에서 연속된 공백을 제거하고 tibble 구조로 변환한 다음
# 문장 기준으로 토큰화하세요.
# Q5. 연설문에서 "경제"가 사용된 문장을 출력하세요


#Q1. speech_park.txt를 불러와 분석에 적합하게 전처리한 다음 연설문에서 명사를 추출하세요.
raw_park <- readLines("TM-dataset/speech_park.txt", encoding = "UTF-8")

library(dplyr)
library(stringr)
raw_park <- raw_park[raw_park!="\n"]
raw_park <- raw_park[raw_park!=""]

park <- raw_park %>%
  str_replace_all("[^가-힣]", " ") %>% 
  str_squish() %>% 
  as_tibble() 
park

# A tibble: 49 x 1
# value                                       
# <chr>                                       
#   1 존경하는 국민 여러분 저는 오늘 국민 한 분 ~ 
#   2 국민 여러분 저의 삶은 대한민국과 함께 해온 ~
#   3 어머니가 흉탄에 돌아가신 후 견딜 수 없는 고~
#   4 그때부터 제 삶은 완전히 다른 길을 가야했습~ 
#   5 아버지를 잃는 또 다른 고통과 아픔을 겪고 저~
#   6 당이 두 번이나 존폐의 위기를 맞고 국민들의 ~
#   7 저 박근혜 그 동안의 제 삶이 저 혼자만의 삶~ 
#   8 어떤 국민도 홀로 뒤처져 있지 않게 할 것입니~
#   9 국민 여러분 우리는 지금 중요한 기로에 서 있~
#   10 지금 우리 국민들은 불안합니다 청년들은 일자~


library(tidytext)
library(KoNLP)
word_noun <- park %>%
  unnest_tokens(input = value,
                output = word,
                token = extractNoun)
word_noun


# Q2. 가장 자주 사용된 단어 20개를 추출하세요.
top20 <- word_noun %>%
  count(word, sort = T) %>%
  filter(str_count(word) > 1) %>%
  head(20)
top20


#Q3. 가장 자주 사용된 단어 20개의 빈도를 나타낸 막대 그래프를 만드세요.
library(ggplot2)
ggplot(top20, aes(x = reorder(word, n), y = n)) +
  geom_col() +
  coord_flip () +
  geom_text(aes(label = n), hjust = -0.3) +
  labs(x = NULL)


# Q4. 전처리하지 않은 연설문에서 연속된 공백을 제거하고 tibble 구조로 변환한 다음
# 문장 기준으로 토큰화하세요.
sentences_park <- raw_park %>%
  str_squish() %>%
  as_tibble() %>% 
  unnest_tokens(input = value,
                output = sentence,
                token = "sentences")
sentences_park


#Q5. 연설문에서 "경제"가 사용된 문장을 출력하세요.
sentences_park %>%
  filter(str_detect(sentence, "경제"))

