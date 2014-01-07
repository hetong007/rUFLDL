x = matrix(rnorm(12000),ncol=6)
x[,1] = runif(1000,min=0,max=0.1)
x[,2] = x[,1]+5
x[,3] = x[,2]*5
x[,4] = runif(1000,min=0.9,max=1)
x[,5] = x[,4]+5
x[,6] = x[,5]*5
for (i in 1:6)
    x[,i] = (x[,i]-min(x[,i]))/(max(x[,i])-min(x[,i]))

y = as.numeric(x[,1]*x[,4]+x[,2]-x[,6]>0)

model = DeepNetworks(x[1:1000,],y[1:1000])
W = model[[1]]
b = model[[2]]
fitted = t(apply(x[1:1000,],1,ForwardPropagation,W,b,'single'))
pred = t(apply(x[1001:2000,],1,ForwardPropagation,W,b,'single'))

auc(roc(fitted[,ncol(fitted)],as.factor(y[1:1000])))
auc(roc(pred[,ncol(pred)],as.factor(y[1001:2000])))
