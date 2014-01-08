load('MNIST/data/train.rda')
x = as.matrix(train[1:2000,-1])
y = train[1:2000,1]


#classification mission
model = Backpropagation(x[1:1000,],y[1:1000],nodes=1000,
                        mission='classification',
                        alpha=0.1,lambda=0,maxStep=10000)#Time costy
W = model[[1]]
b = model[[2]]
fitted = t(apply(x[1:1000,],1,ForwardPropagation,W,b,'single'))
fitted = apply(fitted,1,which.max)-1
pred = t(apply(x[1001:2000,],1,ForwardPropagation,W,b,'single'))
pred = apply(pred,1,which.max)-1

table(as.factor(fitted),as.factor(y[1:1000]))
sum(diag(table(as.factor(fitted),as.factor(y[1:1000]))))/
    sum(table(as.factor(fitted),as.factor(y[1:1000])))
table(as.factor(pred),as.factor(y[1001:2000]))
sum(diag(table(as.factor(pred),as.factor(y[1001:2000]))))/
    sum(table(as.factor(pred),as.factor(y[1001:2000])))
