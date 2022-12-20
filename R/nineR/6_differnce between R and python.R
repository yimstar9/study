# 6_differnce between R and python
#막대차트(가로,세로)
library(ggplot2)
years <-  c('2018', '2019', '2020')
values <-  c(100, 400, 900)
df<-data.frame(values,years)
ggplot(df,aes(x=years,y=values))+geom_bar(stat = "identity")

ggplot(df,aes(x=years,y=values))+geom_bar(stat = "identity") + coord_flip()


#누적막대 차트
library(dplyr)
library(reshape2)

name <- c("R. Lewandowski","L. Messi","H. Kane","K. Mbappe","E. Haaland","S. HeungMin")
goal <- c(41,30,23,26,27,38)
ass <- c(7,9,15,8,6,19)

df <- data.frame(name,ass,goal)
df <- melt(df)

ggplot(df, aes(y=value, x=name,fill=variable)) + 
  geom_bar(position="stack", stat="identity")

#점차트
data(mpg)
 ggplot() +
   geom_point(mapping=aes(x=displ, y=hwy, color=class), data=mpg,cex=3)

#원형 차트
library(dplyr)
library(ggplot2)
name <- c("R. Lewandowski","L. Messi","H. Kane","K. Mbappe","E. Haaland","S. HeungMin")
goal <- c(41,30,23,26,27,38)
ass <- c(7,9,15,8,6,19)
df <- data.frame(name,ass,goal)
df <- df %>% mutate(percent=(goal/sum(goal))*100)
ggplot(df, aes(x='', y=percent, fill=name))+
  geom_bar(stat='identity')+
  theme_void()+
  coord_polar('y', start=0)+
  geom_text(aes(label=paste0(round(percent,1), '%')),
            position=position_stack(vjust=0.5))

#상자 그래프
mpg

ggplot(mpg) + geom_boxplot(aes(class,hwy))

#히스토그램
ggplot(mpg) + geom_histogram(aes(hwy))

#산점도
ggplot(mpg, aes(cty, hwy)) + geom_point()

#중첩자료 시각화: 동일한 값을 가지는 데이터가 여러개일 경우 처리 하는 방법.
ggplot(mpg, aes(cty,hwy))+ geom_count( show.legend=T) 


#비교 시각화
mtcars
library(ggplot2)
library(ggcorrplot)

corr <- round(cor(mtcars), 1)

ggcorrplot(corr, hc.order = TRUE, 
           #type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           #method="circle", 
           colors = c("tomato2", "white", "springgreen3"), 
           title="Correlogram of mtcars", 
           ggtheme=theme_bw)


#밀도 그래프
library(ggplot2)

ggplot(mpg, aes(cty)) + geom_density(aes(fill=factor(cyl)), alpha=0.8) + 
                        labs(title="밀도 그래프", 
                        # subtitle="City Mileage Grouped by Number of cylinders",
                        # caption="Source: mpg",
                        # x="City Mileage",
                         fill="# Cylinders")
