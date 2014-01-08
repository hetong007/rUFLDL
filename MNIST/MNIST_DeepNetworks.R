load('MNIST/data/train.rda')
x = as.matrix(train[1:2000,-1])
y = train[1:2000,1]


model = DeepNetworks(x,y,nodes_1=500,maxStep_1=1000,sparsity_1=FALSE,
                     nodes_2=500,maxStep_2=3000,sparsity_2=FALSE,
                     alpha=0.1,lambda=3e-3,maxStep_back = 5000)
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
