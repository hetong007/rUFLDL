load('MNIST/data/train.rda')
x = as.matrix(train[1:2000,-1])
y = train[1:2000,1]

#Not working, due to exp overflow......
model = Softmax(x[1:1000,],y[1:1000],0.1,0.01,maxStep=500)

fitted = model[[3]]
pred = apply(model[[4]] %*% t(cbind(1,x[1001:2000,])),2,which.max)-1

table(as.factor(fitted),as.factor(y[1:1000]))
sum(diag(table(as.factor(fitted),as.factor(y[1:1000]))))/
    sum(table(as.factor(fitted),as.factor(y[1:1000])))
table(as.factor(pred),as.factor(y[1001:2000]))
sum(diag(table(as.factor(pred),as.factor(y[1001:2000]))))/
    sum(table(as.factor(pred),as.factor(y[1001:2000])))
