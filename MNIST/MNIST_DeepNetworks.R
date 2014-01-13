load('MNIST/data/train.rda')
x = as.matrix(train[1:2000,-1])
y = train[1:2000,1]

model = DeepNetworks(x[1:1000,],y[1:1000],
                     nodes_1=500,maxStep_1=500,sparsity_1=TRUE,
                     nodes_2=500,maxStep_2=500,sparsity_2=TRUE,maxStep_3=10000,
                     alpha=0.1,lambda=3e-3,maxStep_back = 1000)
W = model[[1]]
b = model[[2]]
fitted = ForwardPropagation(x[1:1000,],W,b,'single')
fitted = t(fitted)
fitted = apply(fitted,1,which.max)-1
pred = ForwardPropagation(x[1001:2000,],W,b,'single')
pred = t(pred)
pred = apply(pred,1,which.max)-1

table(as.factor(fitted),as.factor(y[1:1000]))
sum(diag(table(as.factor(fitted),as.factor(y[1:1000]))))/
    sum(table(as.factor(fitted),as.factor(y[1:1000])))
#expected to be 0.99
table(as.factor(pred),as.factor(y[1001:2000]))
sum(diag(table(as.factor(pred),as.factor(y[1001:2000]))))/
    sum(table(as.factor(pred),as.factor(y[1001:2000])))
#expected to be 0.848
