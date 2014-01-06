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


tmp=ParameterInitializer(c(6,10,1))
W = tmp[[1]]
b = tmp[[2]]

model = Backpropagation(x[1:1000,],y[1:1000],W,b,
                        alpha=0.1,lambda=0,maxStep=10000)
W = model[[1]]
b = model[[2]]
fitted = apply(x[1:1000,],1,ForwardPropagation,W,b,'single')
pred = apply(x[1001:2000,],1,ForwardPropagation,W,b,'single')

require(AUC)
auc(roc(fitted,as.factor(y[1:1000])))
auc(roc(pred,as.factor(y[1001:2000])))


feature = FeatureExtract(x,whitening='ZCA',
                         layers=c(6,2,6),sparsity=FALSE)
tmp=ParameterInitializer(c(2,10,1))
W = tmp[[1]]
b = tmp[[2]]
model = Backpropagation(feature[1:1000,],y[1:1000],W,b,
                        alpha=0.1,lambda=0,maxStep=10000)
W = model[[1]]
b = model[[2]]
fitted = apply(feature[1:1000,],1,ForwardPropagation,W,b,'single')
pred = apply(feature[1001:2000,],1,ForwardPropagation,W,b,'single')

require(AUC)
auc(roc(fitted,as.factor(y[1:1000])))
auc(roc(pred,as.factor(y[1001:2000])))


feature = FeatureExtract(x,whitening='ZCA',
                         layers=c(6,40,6),sparsity=TRUE)
tmp=ParameterInitializer(c(40,10,1))
W = tmp[[1]]
b = tmp[[2]]
model = Backpropagation(feature[1:1000,],y[1:1000],W,b,
                        alpha=0.1,lambda=0,maxStep=2000)
W = model[[1]]
b = model[[2]]
fitted = apply(feature[1:1000,],1,ForwardPropagation,W,b,'single')
pred = apply(feature[1001:2000,],1,ForwardPropagation,W,b,'single')

require(AUC)
auc(roc(fitted,as.factor(y[1:1000])))
auc(roc(pred,as.factor(y[1001:2000])))
