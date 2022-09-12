setwd("C:/Users/Kieran Warwick/Documents/DFGPU/Formal/correctness")


file <- read.csv(file = 'add.csv')
summary(file$epULP)
sd(file$epULP)
pdf("addEP.pdf") 
t <- table(file$epULP)
plot(t / sum(t) * 100,xlim=c(0, 100),ylim=c(0, 15),
  xlab="Error from correct value / ULP",
  ylab="Percentage / %")
dev.off() 

summary(file$floatULP)
sd(file$floatULP)

summary(file$doubleULP)
sd(file$floatULP)

file <- read.csv(file = 'sub.csv')
summary(file$epULP)
sd(file$epULP)
pdf("subEP.pdf") 
t <- table(file$epULP)
plot(t / sum(t) * 100,xlim=c(0, 100),ylim=c(0, 15),
  xlab="Error from correct value / ULP",
  ylab="Percentage / %")
dev.off() 

summary(file$floatULP)
sd(file$floatULP)

summary(file$doubleULP)
sd(file$floatULP)

file <- read.csv(file = 'mul.csv')
summary(file$epULP)
sd(file$epULP)
pdf("mulEP.pdf") 
t <- table(file$epULP)
plot(t / sum(t) * 100,xlim=c(0, 100),ylim=c(0, 15),
  xlab="Error from correct value / ULP",
  ylab="Percentage / %")
dev.off() 

summary(file$floatULP)
sd(file$floatULP)

summary(file$doubleULP)
sd(file$floatULP)

file <- read.csv(file = 'div.csv')
summary(file$epULP)
sd(file$epULP)
pdf("divEP.pdf") 
t <- table(file$epULP)
plot(t / sum(t) * 100,xlim=c(0, 100),ylim=c(0, 15),
  xlab="Error from correct value / ULP",
  ylab="Percentage / %")
dev.off()

summary(file$floatULP)
sd(file$floatULP)

summary(file$doubleULP)
sd(file$floatULP)