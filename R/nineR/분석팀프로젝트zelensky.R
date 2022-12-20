#zelensky_address 분석
library(KoNLP)
library(tm)
library(wordcloud)
library(wordcloud2)
library(dplyr)

user_dic <- data.frame(term = c("들"), tag = 'ncn')
buildDictionary(ext_dic = 'sejong', user_dic = user_dic)

 
# wordcloud2(df)
# head(df)

######빈줄 제거 reference########
words <- scan("Zelensky_address.txt",what="character")
words2 <- words[words != "\\n"]
#df <- as.data.frame(as.matrix(words2))
#df$num<-1
#df%>%group_by(V1)%>%summarise(count=sum(num))

onepar <- paste(words2, collapse = " ")
onepar
library(stringr)
lines <- str_split(onepar, pattern = fixed(" \\r\\n"), simplify = TRUE)
lines
##########################################################
######빈줄제거##############


#################################################
data <- readLines("Zelensky_address.txt")
data<-data[data!=""]
data <- data[data != "\\n"]
str(data)
exNouns <- function(x) {paste(extractNoun(as.character(x)), collapse = " ") }
#exNouns <- function(x) {(extractNoun(as.character(x))) }
data_nouns <- sapply(data, exNouns)
data_nouns
mywords <- Corpus(VectorSource(data_nouns))
mywords <- tm_map(mywords,removePunctuation)
mywords <- tm_map(mywords,removeNumbers)
mywords <- tm_map(mywords,tolower)
mywords <- tm_map(mywords, removeWords, stopwords('english'))
mywords <- tm_map(mywords, removeWords, stopwords('SMART'))
inspect(mywords)
# 실습: 단어 선별(2 ~ 8 음절 사이 단어 선택)하기
# 단계 1: 전처리된 단어집에서 2 ~ 8 음절 단어 대상 선정
#help("TermDocumentMatrix")
myCorpusPrepro_term <- TermDocumentMatrix(mywords,control = list(wordLengths = c(4, 16)))
myCorpusPrepro_term
# 단계 2: matrix 자료구조를 data.frame 자료구조로 변경
myTerm_df <- as.data.frame(as.matrix(myCorpusPrepro_term))
dim(myTerm_df)


# 실습: 단어 출현 빈도수 구하기
wordResult <- sort(rowSums(myTerm_df), decreasing = TRUE)
wordResult[1:10]
wordResult

myName <- names(wordResult)
word.df <- data.frame(word = myName, freq = wordResult)
  
pal <- brewer.pal(12, "Paired")
wordcloud(word.df$word, word.df$freq, scale = c(5, 1),
          min.freq = 3, random.order = F,
          rot.per = .1, colors = pal, family = "malgun")

wordcloud2(word.df)

#############################################################
#방법2
#https://signedinfo.com/entry/%EC%9E%90%EC%97%B0%EC%96%B4-%EC%B2%98%EB%A6%AC
test <- SimplePos09("대화하자")
test
data <- readLines("Zelensky_address.txt")
#data<-data[data!=""]
sdata<-SimplePos09(data)
sdata[[]]
unlist(sdata)
grep("/N$|/N+",unlist(sdata),value=TRUE)
noun_ext=function(x){
  x=grep("/N$|/N+|/P$|/P+",unlist(x),value=TRUE)
  return(x)
}
result <- lapply(sdata,FUN="noun_ext")

noun_ext2=function(x){
  x=grep("/N$|/N+|/P$|/P+",unlist(x),value=TRUE)
  x=as.character(gsub("/[A-Z].*?$","",x))
  return(x)
}
result <- lapply(sdata,noun_ext2)

result <- unlist(result)

result <- Filter(function(x){  nchar(x) >=2& nchar(x)<10} , result) # (2~10)글자 이상되는것만 필터링
result

df <- as.data.frame(table(result))
df%>%arrange(-Freq)%>%head(20)

wordcloud2(df,size=2,minSize = 1,gridSize =25,shape="triangle",shuffle = T)


##############################연관어 분석##########################
library(arules)
library(KoNLP)
library(backports) 

data <- readLines("Zelensky_address.txt")
lword <- Map(extractNoun, data)
lword <- unique(lword)
length(lword)

lword <- sapply(lword, unique)

filter1 <- function(x) {
  nchar(x) <= 4 && nchar(x) >= 2 && is.hangul(x)
}
filter2 <- function(x) { Filter(filter1, x) }


lword <- sapply(lword, filter2)

wordtran <- as(lword, "transactions")
wordtran

tranrules <- apriori(wordtran, parameter = list(supp = 0.02, conf = 0.1))
summary(tranrules)
arules::inspect(tranrules)

#############################################연관어 시각화하기################

# 단계 1: 연관단어 시각화를 위해서 자료구조 변경
rules <- labels(tranrules, ruleSep = " ")
rules

# 단계 2: 문자열로 묶인 연관 단어를 행렬구조로 변경
rules <- sapply(rules, strsplit, " ", USE.NAMES = F)
rules
# 단계 3: 행 단위로 묶어서 matrix로 변환
rulemat <- do.call("rbind", rules)
class(rulemat)
# 단계 4: 연관어 시각화를 위한 igraph 패키지 설치와 로딩
# install.packages("igraph")
# install.packages("igraph", type=igraph)
#install.packages(‘igraph’, type=’binary’)
library(igraph)
# 단계 5: edgelist 보기
rulemat
ruleg <- graph.edgelist(rulemat[c(3:127), ], directed = F)
ruleg
# 단계 6: edgelist 시각화
dev.off()
plot.igraph(ruleg, vertex.label = V(ruleg)$name,
            vertex.label.cex = 1.2, vertext.label.color = 'black',
            vertex.size = 12, vertext.color = 'green',
            vertex.frame.co.or = 'blue')
