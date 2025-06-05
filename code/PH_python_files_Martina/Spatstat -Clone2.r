
library(spatstat)

# Completely spatially random in window of [0,1]x[0,1]
x <- runif(20)
y <- runif(20)
X <- ppp(x, y, c(0,1), c(0,1))
plot(X)

#other point patterns
#matern cluster process
X <- rMaternI(20, 0.05)

#X<-rcell(nx=10)
plot(X)

#Put in your point pattern model which is not just completely random
number_pt_patterns=10
number_points_in_patterns=integer(number_pt_patterns)
for (i in 0:9){
    X <- rMaternI(40, 0.05)
 #   plot(X)
    print(X)
    data=c(X$x,X$y)
    number_points_in_patterns[i+1]=length(X$x)
    B = matrix(data,nrow=2)
    pointlist=t(B)
    name=paste("pointpattern",i,".csv", sep = "")
    print(name)
    write.csv(pointlist, file =name )}

print(number_points_in_patterns)

for (i in 0:9){
    print(i)
    print(i+number_pt_patterns)
    size=number_points_in_patterns[i+1]
    print(size)
    x <- runif(size)
    y <- runif(size)
    #X <- ppp(x, y, c(0,1), c(0,1))
    #plot(X)
    data=c(x,y)
    B = matrix(data,nrow=2)
    pointlist=t(B)
    name=paste("pointpattern",i+number_pt_patterns,".csv", sep = "")
    print(name)
    write.csv(pointlist, file =name )}


#X <- rMaternI(20, 0.05)
#print(X$x)
#print(X$y)
data=c(X$x,X$y)
B = matrix(data,nrow=2)  
#print(B)
pointlist=t(B)
write.csv(pointlist, file = "pointpattern.csv")
#this writes the point pattern as a list of points in a csv file (each line a different point)
