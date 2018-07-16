##Analysis on Wine data

#seeds - http://archive.ics.uci.edu/ml/datasets/seeds
mydata <- read.csv("seeds.csv", header=TRUE)
#Remove the class variable if exists
mydata <- mydata[,1:7]
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))

for (i in 2:15) 
  wss[i] <- sum(kmeans(mydata, centers=i)$withinss)

plot(1:15, wss, type="b", xlab="Number of Clusters",ylab="Within groups sum of squares")


#Wine  - https://archive.ics.uci.edu/ml/datasets/Wine
mydata <- read.csv("Wine.csv", header=TRUE)
#Remove the class variable if exists
mydata <- mydata[,2:14]
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))

for (i in 2:15) 
  wss[i] <- sum(kmeans(mydata, centers=i)$withinss)

plot(1:15, wss, type="b", xlab="Number of Clusters",ylab="Within groups sum of squares")

#BreastTissue - https://archive.ics.uci.edu/ml/datasets/Breast+Tissue
mydata <- read.csv("resources//BreastTissue.csv",header=TRUE)
#Remove the class variable if exists
mydata <- mydata[,1:9]
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))

for (i in 2:15) 
  wss[i] <- sum(kmeans(mydata, centers=i)$withinss)

plot(1:15, wss, type="b", xlab="Number of Clusters",ylab="Within groups sum of squares")