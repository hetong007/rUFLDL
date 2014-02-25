load('MNIST/data/train.rda')

tr_ind = 1:1000
te_ind = 1001:2000

ind = c(tr_ind,te_ind)
x = as.matrix(train[ind,-1])
y = train[ind,1]

model = DeepNetworks(x[tr_ind,],y[tr_ind],
                     nodes_1=500,maxStep_1=500,sparsity_1=TRUE,
                     nodes_2=500,maxStep_2=500,sparsity_2=TRUE,maxStep_3=10000,
                     alpha=0.1,lambda=3e-3,maxStep_back = 1000)
W = model[[1]]
b = model[[2]]
fitted = ForwardPropagation(x[tr_ind,],W,b,'single')
fitted = t(fitted)
fitted = apply(fitted,1,which.max)-1
pred = ForwardPropagation(x[te_ind,],W,b,'single')
pred = t(pred)
pred = apply(pred,1,which.max)-1

table(as.factor(fitted),as.factor(y[tr_ind]))
sum(diag(table(as.factor(fitted),as.factor(y[tr_ind]))))/
    sum(table(as.factor(fitted),as.factor(y[tr_ind])))
#expected to be 0.99
table(as.factor(pred),as.factor(y[te_ind]))
sum(diag(table(as.factor(pred),as.factor(y[te_ind]))))/
    sum(table(as.factor(pred),as.factor(y[te_ind])))
#expected to be 0.848
