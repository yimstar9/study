#임성구
#1. 제공된 데이터에서 빈도수가 2회 이상 단어를 이용하여 단어 구름으로 시각화 하시오
#(1) 텍스트 데이터 가져오기

raw <- file("젤렌스키_연설문_20220219.txt", encoding = "UTF-8")
data <- readLines(raw)
head(data)
close(raw)

#(2) 추출된 단어 전처리
data<-data[data!=""&data!=" "]
str(data)

exNouns <- function(x) {paste(extractNoun(as.character(x)), collapse = " ") }
data_nouns <- sapply(data, exNouns)
data_nouns

mywords <- Corpus(VectorSource(data_nouns))
mywords <- tm_map(mywords,removePunctuation)
mywords <- tm_map(mywords,removeNumbers)
mywords <- tm_map(mywords,tolower)
mywords <- tm_map(mywords, removeWords, stopwords('english'))
mywords <- tm_map(mywords, removeWords, stopwords('SMART'))
inspect(mywords)

myCorpusPrepro_term <-
  TermDocumentMatrix(mywords)
myCorpusPrepro_term

myTerm_df <- as.data.frame(as.matrix(myCorpusPrepro_term))
dim(myTerm_df)

#(3) 단어 출현 빈도 산출
wordResult <- sort(rowSums(myTerm_df), decreasing = TRUE)
wordResult[1:10]
#빈도수 2회 이상 단어 추출
wordResult <- wordResult[wordResult>=2]

#(4) 단어구름에 디자인 적용
myName <- names(wordResult)
word.df <- data.frame(word = myName, freq = wordResult)
size=1.6
mins=2

#(5) wordcloud2 패키지 사용하여 워드클라우드 결과 제출
wordcloud2(word.df,size = size, minSize = mins)


##################################################################
#2. 다음 텍스트를 대상으로 감정분석 하시오
#(1) 단어별로 token화 하시오

library(readr)
library(textclean)
library(dplyr)
library(tidytext)


#데이터 불러오기
a <- file("Itaewon_text1.txt", encoding = "UTF-8")
b <- readLines(a)
b<-b[b!=""]
b<-b[1]
close(a)
head(b)
#한줄로 된 데이터를 문장으로 분리하기
b<-strsplit(b,fixed=TRUE,split=". ")
b<-unlist(b)
df <- tibble(sentence =b)
df

# 토큰화
df <- df %>%unnest_tokens(input = sentence,
                          output = word,
                          token = "words",
                          drop = F)
df %>% print(n = Inf)
df<-df %>%
  select(word,sentence)

################################
#(2) 문장별 감성점수를 산출하시오

# 감정 사전 불러오기
dic <- read_csv("TM-dataset/knu_sentiment_lexicon.csv")

#감정 사전 확인

dic %>%
  mutate(sentiment = ifelse(polarity >= 1, "pos",
                            ifelse(polarity <= -1, "neg", "neu"))) %>%
  count(sentiment)

#문장별 감성점수 
word_comment <- df %>%
  left_join(dic, by = "word") %>%
  mutate(polarity = ifelse(is.na(polarity), 0, polarity))
print(word_comment,n=100)

score_ <- word_comment %>%
  group_by(sentence) %>%
  summarise(score = sum(polarity))%>%ungroup()
score_ %>%
  select(score, sentence)

# # A tibble: 3 x 2
# score sentence                                                         
# <dbl> <chr>                                                            
#   1    -1 20대 아들을 키우는 입장에서 이 참사를 보고 가만히 있을 수 없었다~
#   2     0 시민들은 국가가 참사를 방치했다고 입을 모았다                    
#   3     0 집회 시작 30분 전부터 눈물을 흘리고 있던 이용신씨(55)는 “세월호~ 
