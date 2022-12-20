# 
# 다차원 척도법(Multi-dimensional Scaling)
# 여러 대상의 특징 사이 관계에 대한 수치적 자료를 이용하여 유사성에 대한 측정치를
# 상대적 거리로 구조화하는 방법
# 2차원 또는 3차원에서의 특정 위치에 관측치를 배치해서 보기 쉽게 척도화
# 즉, 항목 사이 거리를 기준으로 하는 자료를 이용하여 항목들의 상대적인 위치를 찾고
# 거리가 가까운 개체들끼리 Group 화 하여 분류할 수 있다.
# 다차원 척도법 적용 절차
# 1) 자료 수집: 특성을 측정
# 2) 유사성, 비유사성 측정: 개체 사이의 거리 측정
# 3) 공간에서 개체 사이 거리 표현
# 4) 개체의 상호 위치에 따른 관계가 개체들 사이 비유사성에 적합여부 결정
# 다차원 척도법의 종류
# 1) 계량적(전통적) 다차원 척도법(Classical MDS)
# 숫자 데이터로만 구성. 
# stats패키지의 cmdscale()함수
# https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/cmdscale
# 
# 실습
# eurodist 데이터를 이용하여 유럽 주요 도시 사이의 거리 대상으로 다차원 척도법을
# 적용하여 그래프로 표현
# =====================
  # install.packages("MASS")
library(MASS)
data("eurodist")
eurodist
# 다차원 척도법 적용
MDSeurodist <- cmdscale(eurodist)
MDSeurodist
# 시각화
plot(MDSeurodist) 
text(MDSeurodist, rownames(MDSeurodist), cex=0.8, col="blue")
abline(v=0, h=0, lty=1, lwd=0.5)
# ===========================
# 
# 그래프를 통해 Paris, Munich 등이 중심에 있고 Lisbon, Stockholm, Athem 등은 중심에서
# 거리가 좀 있음을 알 수 있다.
# 이 그래프를 통해서 거리가 가까운 도시끼리 Group화가 가능하다.

# 2) 비계량적 다차원 척도법 (nonmetric MDS): 
#   숫자가 아닌 데이터 포함. 
# MASS패키지의 isoMDS()함수 이용
# https://www.rdocumentation.org/packages/MASS/versions/7.3-53.1/topics/isoMDS
# 실습.
# HSAUR 패키지 내 voting 데이터를 이용
# 15명의 의뭔이 19개의 법안에 투표한 결과 데이터
# ================
  # install.packages("HSAUR")
library(HSAUR)
library(MASS)
data("voting", package="HSAUR")
voting
MDS2voting <- isoMDS(voting) 
MDS2voting 
x <- MDS2voting$point[,1]
y <- MDS2voting$point[,2]
plot(x,y) 
text(x, y, labels= colnames(voting))
# =================
# 그래프를 통해 유사한 성향의 의원을 파악하기 쉽다.
# 다차원 척도법은 주어진 데이터를 기반으로 수행하는 일종의 군집 분석이라고 할 수 있다.

