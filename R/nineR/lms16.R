#임성구

# 1. 제공된 데이터에서 빈도수가 2회 이상 단어를 이용하여 단어 구름으로 시각화 하시오
# 
# (1) 텍스트 데이터 가져오기
# (2) 추출된 단어 대상 전처리
# (3) 단어 출현 빈도수 산출
# (4) 단어 구름에 디자인 적용 (wordcloud2 패키지 사용)
# (5) wordcloud2 패키지 사용하여 워드클라우드 결과 제출

library(KoNLP)
library(tm)
library(wordcloud2)
#(1) 텍스트데이터 가져오기
facebook <- file("dataset3/facebook_bigdata.txt", encoding = "UTF-8")
facebook_data <- readLines(facebook)
head(facebook_data)
close(facebook)

exNouns <- function(x) {paste(extractNoun(as.character(x)), collapse = " ") }
facebook_nouns <- sapply(facebook_data, exNouns)

#(2) 추출된 단어 대상 전처리
mywords <- Corpus(VectorSource(facebook_nouns))
mywords <- tm_map(mywords,removePunctuation)
mywords <- tm_map(mywords,removeNumbers)
mywords <- tm_map(mywords,tolower)
mywords <- tm_map(mywords, removeWords, stopwords('english'))
mywords <- tm_map(mywords, removeWords, stopwords('SMART'))
inspect(mywords)

myCorpusPrepro_term <- TermDocumentMatrix(mywords)
myCorpusPrepro_term

myTerm_df <- as.data.frame(as.matrix(myCorpusPrepro_term))
dim(myTerm_df)

#(3) 단어 출현 빈도 산출
wordResult <- sort(rowSums(myTerm_df), decreasing = TRUE)
wordResult[1:10]
wordResult <- wordResult[wordResult>=2]

#(4) 단어구름에 디자인 적용
myName <- names(wordResult)
word.df <- data.frame(word = myName, freq = wordResult)
size=2
mins=2


#(5) wordcloud2 패키지 사용하여 워드클라우드 결과 제출
wordcloud2(word.df,size = size, minSize = mins)


# 2. 다음 텍스트를 대상으로 감성분석을 실시하시오.
# 
# Itaewon_text.txt
# 
# (1)	단어별로 token화 하시오
# (2)	문장별 감성점수를 산출하시오.
library(readr)
library(textclean)
library(dplyr)

a <- file("itaewon.txt", encoding = "UTF-8")
b <- readLines(a)
b<-b[b!=""]
close(a)
head(b)
df <- tibble(sentence =b)
df
# 토큰화
df <- df %>%unnest_tokens(input = sentence,
                output = word,
                token = "words",
                drop = F)
df %>% print(n = Inf)
df %>%
  select(word,sentence)
# 감정 점수 부여
word_comment <- df %>%
  left_join(dic, by = "word") %>%
mutate(polarity = ifelse(is.na(polarity), 0, polarity))

# -------------------------------------------------------------------------
score_ <- word_comment %>%
  group_by(sentence) %>%
  summarise(score = sum(polarity))%>%ungroup()
score_ %>%
  select(score, sentence)
