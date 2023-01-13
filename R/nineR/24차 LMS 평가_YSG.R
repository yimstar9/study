# (의미망 분석)
# 제공된 데이터를 대상으로 의미망 분석을 실행하시오.
# 
# 데이터: news_comment_BTS.csv
# 
# (데이터 loading)
# (1) 대상 데이터 읽어오기
library(readr)
raw_news_comment <- read_csv("TM-dataset/news_comment_BTS.csv")

# # A tibble: 1,200 x 5
# reg_time            reply    
# <dttm>              <chr>    
#  1 2020-09-01 22:58:09 "국보소~ 
#  2 2020-09-01 09:56:46 "아줌마~ 
#  3 2020-09-01 09:08:06 "팩트체~ 
#  4 2020-09-01 08:52:32 "방탄소~ 
#  5 2020-09-01 08:36:33 "대단한 ~
#  6 2020-09-01 08:34:14 "정국오~ 
#  7 2020-09-01 08:32:14 "정말 축~
#  8 2020-09-01 08:22:09 "기자는 ~
#  9 2020-09-01 08:17:58 "자랑스~ 
# 10 2020-09-01 08:15:37 "SuperM ~
#   # ... with 1,190 more rows, and
#   #   3 more variables:
#   #   press <chr>, title <chr>,
#   #   url <chr>
#   # i Use `print(n = ...)` to see more rows, and `colnames()` to see all variable names



##################################################################
# [문항2]  (텍스트 데이터 전처리)
# (2) 전처리를 실행(국문 추출, 중복공백 제거)
library(dplyr)
library(stringr)
library(textclean)
news_comment <- raw_news_comment %>% 
  select(reply) %>%
  mutate(reply = str_replace_all(reply, "[^가-힣]", " "), 
         reply = str_squish(reply),
         id = row_number())

# # A tibble: 1,200 x 2
# reply                      id
# <chr>                   <int>
#   1 국보소년단                  1
# 2 아줌마가 들어도 좋더라      2
# 3 팩트체크 현재 빌보드 ~      3
# 4 방탄소년단이 한국사람~      4
# 5 대단한 월드 클래스는 ~      5
# 6 정국오빠 생일과 더불어~     6
# 7 정말 축하하고 응원하지~     7
# 8 기자는 자고 일어났지만~     8
# 9 자랑스럽다 축하합니다       9
# 10 늘 응원하고 사랑합니다     10
# # ... with 1,190 more rows
# # i Use `print(n = ...)` to see more rows


##########################################################
# [문항3]  (텍스트 데이터 토큰화)
# (3) 형태소 분석기를 이용하여 품사 기준 토큰화 실행
library(tidytext)
library(KoNLP)
comment_pos <- news_comment %>%
  unnest_tokens(input = reply, 
                output = word, 
                token = SimplePos22, 
                drop = F)
comment_pos %>%select(reply, word)

# # A tibble: 11,853 x 2
# reply                         word 
# <chr>                         <chr>
#   1 국보소년단                    국보~
#   2 국보소년단                    단/ma
#   3 아줌마가 들어도 좋더라        아줌~
#   4 아줌마가 들어도 좋더라        들/p~
#   5 아줌마가 들어도 좋더라        좋/p~
#   6 아줌마가 들어도 좋더라        라/nc
#   7 팩트체크 현재 빌보드 위 방탄~ 팩트~
#   8 팩트체크 현재 빌보드 위 방탄~ 현재~
#   9 팩트체크 현재 빌보드 위 방탄~ 빌보~
#   10 팩트체크 현재 빌보드 위 방탄~ 위/nc
# # ... with 11,843 more rows
# # i Use `print(n = ...)` to see more rows



#####################################################
# [문항4]  (텍스트 데이터 품사 분리)
# (4) 품사를 분리하여 행 구성
library(tidyr)
comment_pos <- comment_pos %>% 
  separate_rows(word, sep = "[+]")
comment_pos %>%
  select(word, reply)

# # A tibble: 20,851 x 2
# word        reply                 
# <chr>       <chr>                 
# 1 국보소년/nc 국보소년단            
# 2 단/ma       국보소년단            
# 3 아줌마/nc   아줌마가 들어도 좋더라
# 4 가/jc       아줌마가 들어도 좋더라
# 5 들/pv       아줌마가 들어도 좋더라
# 6 어도/ec     아줌마가 들어도 좋더라
# 7 좋/pa       아줌마가 들어도 좋더라
# 8 더/ep       아줌마가 들어도 좋더라
# 9 어/ec       아줌마가 들어도 좋더라
# 10 라/nc       아줌마가 들어도 좋더라
# # ... with 20,841 more rows
# # i Use `print(n = ...)` to see more rows


######################################################
# [문항5]  (텍스트 데이터 빈도 계산)
# (5) 명사를 추출하여 명사의 빈도를 계산하시오.
noun <- comment_pos %>%
  filter(str_detect(word, "/n")) %>%
  mutate(word = str_remove(word, "/.*$")) 

noun %>%
  count(word, sort = T)

# # A tibble: 2,746 x 2
# word           n
# <chr>      <int>
# 1 위           193
# 2 진짜         179
# 3 자랑         138
# 4 방탄         134
# 5 빌보드       131
# 6 방탄소년단   116
# 7 것            91
# 8 축하          90
# 9 나            81
# 10 축하해        78
# # ... with 2,736 more rows
# # i Use `print(n = ...)` to see more rows



#######################################################
# [문항6]  (텍스트 데이터 결합)
# (6) 동사, 형용사를 추출하고 추출한 데이터를 결합
# (두 글자 이상 단어만 추출)
pvpa <- comment_pos %>%
  filter(str_detect(word, "/pv|/pa")) %>%        # "/pv", "/pa" 추출
  mutate(word = str_replace(word, "/.*$", "다")) # "/"로   시작   문자를 "다"로   바꾸기 

comment <- bind_rows(noun, pvpa) %>% 
  filter(str_count(word) >= 2) %>% 
  arrange(id)
comment %>%
  select(word, reply)

# # A tibble: 7,539 x 2
# word       reply                   
# <chr>      <chr>                   
# 1 국보소년   국보소년단              
# 2 아줌마     아줌마가 들어도 좋더라  
# 3 들다       아줌마가 들어도 좋더라  
# 4 좋다       아줌마가 들어도 좋더라  
# 5 팩트체크   팩트체크 현재 빌보드 위~
# 6 빌보드     팩트체크 현재 빌보드 위~
# 7 방탄소년단 팩트체크 현재 빌보드 위~
# 8 방탄소년단 방탄소년단이 한국사람이~
# 9 한국사람   방탄소년단이 한국사람이~
# 10 자랑       방탄소년단이 한국사람이~
# # ... with 7,529 more rows
# # i Use `print(n = ...)` to see more rows



#######################################################
# [문항7]  (텍스트 데이터 동시 출현 빈도 계산)
# (7) 단어 동시 출현 빈도를 계산하시오.
library(widyr) 
pair <- comment %>%
  pairwise_count(item = word, 
                 feature = id, 
                 sort = T) 
pair

# # A tibble: 73,776 x 3
# item1      item2          n
# <chr>      <chr>      <dbl>
# 1 하다       축하          41
# 2 축하       하다          41
# 3 진짜       방탄소년단    28
# 4 방탄소년단 진짜          28
# 5 자랑       방탄소년단    27
# 6 방탄소년단 자랑          27
# 7 방탄소년단 빌보드        25
# 8 빌보드     방탄소년단    25
# 9 방탄       진짜          24
# 10 진짜       방탄          24
# # ... with 73,766 more rows
# # i Use `print(n = ...)` to see more rows



#######################################################
# [문항8]  (텍스트 데이터 네트워크 그래프 데이터로 변환)
# (8) 동시 출현 빈도 데이터를 ‘네트워크 그래프 데이터’로 변환하기
# (15회 이상 사용된 단어를 추출하여 생성)
library(tidygraph)
graph_comment <- pair %>% 
  filter(n >= 15) %>%
  as_tbl_graph() 
graph_comment

# # A tbl_graph: 14 nodes and 38 edges
# #
# # A directed simple graph with 3 components
# #
# # Node Data: 14 x 1 (active)
# name      
# <chr>     
# 1 하다      
# 2 축하      
# 3 진짜      
# 4 방탄소년단
# 5 자랑      
# 6 빌보드    
# # ... with 8 more rows
# #
# # Edge Data: 38 x 3
# from    to     n
# <int> <int> <dbl>
# 1     1     2    41
# 2     2     1    41
# 3     3     4    28
# # ... with 35 more rows


#########################################################
# [문항9]  (텍스트 데이터 네트워크 그래프 생성)
# (9) 네트워크 그래프 생성하기
library(ggraph) 
set.seed(1234)

ggraph(graph_comment, layout = "fr") +
  geom_edge_link(color = "gray50", 
                 alpha = 0.5) + 
  geom_node_point(color = "lightcoral", 
                  size = 5) +           
  geom_node_text(aes(label = name),     
                 repel = T, 
                 size = 5,
                 family = "nanumgothic") +
  theme_graph()

############################################################
# (네트워크 그래프 저장)
# (10) 네트워크 그래프를 이미지 파일로 저장하고 제출하시오.
ggsave("24차 LMS 평가_YSG.jpg")
