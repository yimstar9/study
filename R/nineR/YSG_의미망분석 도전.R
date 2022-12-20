#임성구
#Q1. "news_comment_BTS.csv"를 불러온 다음 행 번호를 나타낸 변수를 추가하고 분석에 적합하게
# 전처리하세요.

library(readr)
library(dplyr)
raw_news_comment <- read_csv("/news_comment_BTS.csv")
glimpse(raw_news_comment)
library(stringr)
library(textclean)
news_comment <- raw_news_comment %>%
  select(reply) %>%
  mutate(id = row_number(),
         reply = str_replace_all(reply, "[^가-힣]", " "),
         reply = str_squish(reply))
news_comment %>%
  select(id, reply)
# 품사 기준 토큰화
library(tidytext)
library(KoNLP)
comment_pos <- news_comment %>%
  unnest_tokens(input = reply,
                output = word,
                token = SimplePos22,
                drop = F)

#Q2. 댓글에서 명사, 동사, 형용사를 추출하고 '/로 시작하는 모든 문자'를 '다'로 바꾸세요
# 한 행이 한 품사를 구성하도록 분리
library(tidyr)
comment_pos <- comment_pos %>%
  separate_rows(word, sep = "[+]")
comment_pos %>%
  select(word, reply)

# 명사, 동사, 형용사 추출
comment <- comment_pos %>%
  separate_rows(word, sep = "[+]") %>%
  filter(str_detect(word, "/n|/pv|/pa")) %>%
  mutate(word = ifelse(str_detect(word, "/pv|/pa"),
                       str_replace(word, "/.*$", "다"),
                       str_remove(word, "/.*$"))) %>%
  filter(str_count(word) >= 2) %>%
  arrange(id)

comment %>%
  select(word, reply)

# Q3. 다음 코드를 이용해 유의어를 통일한 다음 한 댓글이 하나의 행이 되도록 단어를 결합하세요.
# 유의어 통일하기
comment <- comment %>%
  mutate(word = case_when(str_detect(word, "축하") ~ "축하",
                          str_detect(word, "방탄") ~ "자랑",
                          str_detect(word, "대단") ~ "대단",
                          str_detect(word, "자랑") ~ "자랑",
                          T ~ word))
# 
# Q4. 댓글을 바이그램으로 토큰화한 다음 바이그램 단어쌍을 분리하세요.
# 바이그램 토큰화
bigram_comment <- line_comment %>%
  unnest_tokens(input = sentence,
                output = bigram,
                token = "ngrams",
                n = 2)
bigram_comment

# 바이그램 단어쌍 분리
bigram_seprated <- bigram_comment %>%
  separate(bigram, c("word1", "word2"), sep = " ")
bigram_seprated

# 
# Q5. 단어쌍 빈도를 구한 다음 네트워크 그래프 데이터를 만드세요.
# 난수를 고정한 다음 네트워크 그래프 데이터를 만드세요.
# 빈도가 3 이상인 단어쌍만 사용하세요.
# 연결 중심성과 커뮤니티를 나타낸 변수를 추가하세요.

# 단어쌍 빈도 구하기
pair_bigram <- bigram_seprated %>%
  count(word1, word2, sort = T) %>%
  na.omit()
pair_bigram

# 네트워크 그래프 데이터 만들기
library(tidygraph)
set.seed(1234)
graph_bigram <- pair_bigram %>%
  filter(n >= 3) %>%
  as_tbl_graph(directed = F) %>%
  mutate(centrality = centrality_degree(),
         group = as.factor(group_infomap()))
graph_bigram



# Q6. 바이그램을 이용해 네트워크 그래프를 만드세요.
# 난수를 고정한 다음 네트워크 그래프를 만드세요.
# 레이아웃을 "fr"로 설정하세요.
# 연결 중심성에 따라 노드 크기를 정하고, 커뮤니티별로 노드 색깔이 다르게 설정하세요.
# 노드의 범례를 삭제하세요.
# 텍스트가 노드 밖에 표시되게 설정하고, 텍스트의 크기를 5로 설정하세요.
library(ggraph)
set.seed(1234)
ggraph(graph_bigram, layout = "fr") +
  geom_edge_link() +
  geom_node_point(aes(size = centrality,
                      color = group),
                  show.legend = F) +
  geom_node_text(aes(label = name),
                 repel = T,
                 size = 5) +
  theme_graph()

# 그래프 꾸미기
library(showtext)
font_add_google(name = "Nanum Gothic", family = "nanumgothic")
set.seed(1234)
ggraph(graph_bigram, layout = "fr") + # 레이아웃
  geom_edge_link(color = "gray50", # 엣지 색깔
                 alpha = 0.5) + # 엣지 명암
  geom_node_point(aes(size = centrality, # 노드 크기
                      color = group), # 노드 색깔
                  show.legend = F) + # 범례 삭제
  scale_size(range = c(4, 8)) + # 노드 크기 범위
  geom_node_text(aes(label = name), # 텍스트 표시
                 repel = T, # 노드밖 표시
                 size = 5, # 텍스트 크기
                 family = "nanumgothic") + # 폰트
  theme_graph() # 배경 삭제
