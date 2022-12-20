# 임성구
# 토픽모델링 분석 도전
# speeches_roh.csv에는 노무현 전 대통령의 연설문 780개가 들어있습니다. speeches_roh.csv를 이용
# 해 문제를 해결해 보세요.
# Q1. speeches_roh.csv를 불러온 다음 연설문이 들어있는 content를 문장 기준으로 토큰화하세요.
# Q2. 문장을 분석에 적합하게 전처리한 다음 명사를 추출하세요.
# Q3. 연설문 내 중복 단어를 제거하고 빈도가 100회 이하인 단어를 추출하세요.
# Q4. 추출한 단어에서 다음의 불용어를 제거하세요.
# stopword <- c("들이", "하다", "하게", "하면", "해서", "이번", "하네",
#               "해요", "이것", "니들", "하기", "하지", "한거", "해주",
#               "그것", "어디", "여기", "까지", "이거", "하신", "만큼")
# Q5. 연설문별 단어 빈도를 구한 다음 DTM을 만드세요.
# Q6. 토픽 수를 2~20개로 바꿔가며 LDA 모델을 만든 다음 최적 토픽 수를 구하세요. 80 / 95
# Q7. 토픽 수가 9개인 LDA 모델을 추출하세요.
# Q8. LDA 모델의 beta를 이용해 각 토픽에 등장할 확률이 높은 상위 10개 단어를 추출한 다음
# 토픽별 주요 단어를 나타낸 막대 그래프를 만드세요.
# Q9. LDA 모델의 gamma를 이용해 연설문 원문을 확률이 가장 높은 토픽으로 분류하세요.
# Q10. 토픽별 문서 수를 출력하세요.
# Q11. 문서가 가장 많은 토픽의 연설문을 gamma가 높은 순으로 출력하고 내용이 비슷한지 살펴보세
# 요.

# Q1. speeches_roh.csv를 불러온 다음 연설문이 들어있는 content를 문장 기준으로 토큰화하세요.

# 연설문 불러오기
useNIADic()
library(readr)
speeches_raw <- read_csv("TM-dataset//speeches_roh.csv")
# 문장 기준 토큰화
library(dplyr)
library(tidytext)
speeches <- speeches_raw %>%
  unnest_tokens(input = content,
                output = sentence,
                token = "sentences",
                drop = F)

# Q2. 문장을 분석에 적합하게 전처리한 다음 명사를 추출하세요.
# 전처리
library(stringr)
speeches <- speeches %>%
  mutate(sentence = str_replace_all(sentence, "[^가-힣]", " "),
         sentence = str_squish(sentence))
# 명사 추출
library(tidytext)
library(KoNLP)
library(stringr)
nouns_speeches <- speeches %>%
  unnest_tokens(input = sentence,
                output = word,
                token = extractNoun,
                drop = F) %>%
  filter(str_count(word) > 1)

# Q3. 연설문 내 중복 단어를 제거하고 빈도가 100회 이하인 단어를 추출하세요.
# 연설문 내 중복 단어 제거
nouns_speeches <- nouns_speeches %>%
  group_by(id) %>%
  distinct(word, .keep_all = T) %>%
  ungroup()
# 단어 빈도 100회 이하 단어 추출
nouns_speeches <- nouns_speeches %>%
  add_count(word) %>%
  filter(n <= 100) %>%
  select(-n)

# Q4. 추출한 단어에서 다음의 불용어를 제거하세요.
stopword <- c("들이", "하다", "하게", "하면", "해서", "이번", "하네",
              "해요", "이것", "니들", "하기", "하지", "한거", "해주",
              "그것", "어디", "여기", "까지", "이거", "하신", "만큼")
# 불용어 제거
nouns_speeches <- nouns_speeches %>%
  filter(!word %in% stopword)

# Q5. 연설문별 단어 빈도를 구한 다음 DTM을 만드세요.
# 연설문별 단어 빈도 구하기
count_word_doc <- nouns_speeches %>%
  count(id, word, sort = T)
# DTM 만들기
dtm_comment <- count_word_doc %>%
  cast_dtm(document = id, term = word, value = n)

# Q6. 토픽 수를 2~20개로 바꿔가며 LDA 모델을 만든 다음 최적 토픽 수를 구하세요.
# 토픽 수 바꿔가며 LDA 모델 만들기
install.packages("ldatuning",type="binary")
library(ldatuning)
models <- FindTopicsNumber(dtm = dtm_comment,
                           topics = 2:20,
                           return_models = T,
                           control = list(seed = 1234))
# 최적 토픽 수 구하기
FindTopicsNumber_plot(models)

# Q7. 토픽 수가 9개인 LDA 모델을 추출하세요.
lda_model <- models %>%
  filter (topics == 9) %>%
  pull(LDA_model) %>%
  .[[1]]

# Q8. LDA 모델의 beta를 이용해 각 토픽에 등장할 확률이 높은 상위 10개 단어를 추출한 다음
# 토픽별 주요 단어를 나타낸 막대 그래프를 만드세요.
# beta 추출
help(tidy)
tidy(lda_model, matrix = "beta")
term_topic <- tidy(lda_model, matrix = "beta")
# 토픽별 beta 상위 단어 추출
top_term_topic <- term_topic %>%
  group_by(topic) %>%
  slice_max(beta, n = 10)
top_term_topic

# 막대 그래프 만들기
library(ggplot2)
ggplot(top_term_topic,
       aes(x = reorder_within(term, beta, topic),
           y = beta,
           fill = factor(topic))) +
  geom_col(show.legend = F) +
  facet_wrap(~ topic, scales = "free", ncol = 3) +
  coord_flip () +
  scale_x_reordered() +
  labs(x = NULL)

# Q9. LDA 모델의 gamma를 이용해 연설문 원문을 확률이 가장 높은 토픽으로 분류하세요.
# gamma 추출
doc_topic <- tidy(lda_model, matrix = "gamma")
# 문서별로 확률이 가장 높은 토픽 추출
doc_class <- doc_topic %>%
  group_by(document) %>%
  slice_max(gamma, n = 1)
# 변수 타입 통일
doc_class$document <- as.integer(doc_class$document)
# 연설문 원문에 확률이 가장 높은 토픽 번호 부여
speeches_topic <- speeches_raw %>%
  left_join(doc_class, by = c("id" = "document"))

# Q10. 토픽별 문서 수를 출력하세요.
speeches_topic %>%
  count(topic)

# Q11. 문서가 가장 많은 토픽의 연설문을 gamma가 높은 순으로 출력하고 내용이 비슷한지 살펴보세
# 요.
speeches_topic %>%
  filter(topic == 9) %>%
  arrange(-gamma) %>%
  select(content)

