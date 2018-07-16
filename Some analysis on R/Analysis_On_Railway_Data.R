##Analysis of Railway Data
##Import the data from CSV
View(dataset3)
 with(dataset3, mean(dataset1.delay[dataset1.minutes==0]))

 reducedDataset <- data.frame(average_delay= numeric(0), minutes= integer(0))
 View(reducedDataset)
 rbind(reducedDataset, c(7.16, 0))
 
View(reducedDataset)
reducedDataset[nrow(reducedDataset) + 1, ] <- c(7.16, 0)
View(reducedDataset)


#Cluster Analysis
 for (minute in 1:59){avg_delay <- with(dataset3, mean(dataset1.delay[dataset1.minutes==minute]))
+     reducedDataset[nrow(reducedDataset) + 1, ] <- c(avg_delay, minute)
+ }
 results1<- kmeans(reducedDataset,3)
 plot(reducedDataset[c("minutes", "average_delay")], col=results1$cluster)
 clusplot(reducedDataset, results1$cluster ,color = TRUE)

library(cluster)
 clusplot(reducedDataset, results1$cluster ,color = TRUE)
 dissE <- daisy(reducedDataset)
 sk <- silhouette(results1$cluster, dissE)
 pdf("resource//silhouette_plot.pdf")
 plot(sk)
 dev.off()
 distMatrix1 <- dist(reducedDataset)
 clustersHierarchical <- hclust(distMatrix1)
 plot(clustersHierarchical)

#Decision tree:
##Import the data from CSV
View(dataset2)
View(dataset1)
View(trains)
 trains_transformed <- transform(trains, delay = ifelse(delay > 0, 1, delay))
 View(trains_transformed)
 trains_input <- trains_transformed[c(1:50000),]
 View(trains_input)
 output_tree <- ctree(delay ~ origin + next_station, data = trains_input)
 plot(output_tree)

#Regression tree:
rpart(formula = delay ~ origin + day, data = trains_input_test)
