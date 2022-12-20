install.packages("devtools")
library(devtools)
#install_github("vqv/ggbiplot") 


data(iris)
head(iris, 3)

log.ir <- log(iris[, 1:4])
ir.species <- iris[, 5]
ir.pca <- prcomp(log.ir, center = TRUE, scale. = TRUE) 

library(ggbiplot)
g <- ggbiplot(ir.pca, obs.scale=1, var.scale=1, groups=ir.species, ellipse=TRUE, circle=TRUE)
g <- g + scale_color_discrete(name='')  
g <- g + theme(legend.direction='horizontal', legend.position='top')
print(g)
 

require(ggplot2)
theta <- seq(0,2*pi,length.out = 100)
circle <- data.frame(x = cos(theta), y = sin(theta))
p <- ggplot(circle,aes(x,y)) + geom_path()
loadings <- data.frame(ir.pca$rotation, .names = row.names(ir.pca$rotation))
p + geom_text(data=loadings, 
                mapping=aes(x = PC1, y = PC2, label = .names, colour = .names)) +
  coord_fixed(ratio=1) +labs(x = "PC1", y = "PC2")
#install.packages("caret")
library(caret)
require(caret) 
trans = preProcess(iris[,1:4], method=c("BoxCox", "center", "scale", "pca")) 
PC = predict(trans, iris[,1:4])

#디폴트로, preProcess() 함수는 적어도 데이터 분산의 95% 이상을 설명하는데 필요한 주
#성분들(PCs)만을 저장(keep)하나, 이것은 thresh=0.95(디폴트) 옵션을 통해 변경될 수 있
#다. 

head(PC, 3)
trans$rotation
