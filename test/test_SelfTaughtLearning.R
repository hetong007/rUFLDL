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

#Primary Backpropagation
model = Backpropagation(x[1:1000,],y[1:1000],node=10,mission='classification',
                        alpha=0.1,lambda=0,maxStep=10000)
W = model[[1]]
b = model[[2]]
fitted = t(ForwardPropagation(x[1:1000,],W,b,'single'))
pred = t(ForwardPropagation(x[1001:2000,],W,b,'single'))

auc(roc(fitted[,ncol(fitted)],as.factor(y[1:1000])))
auc(roc(pred[,ncol(pred)],as.factor(y[1001:2000])))

#Autoencoder
feature = FeatureExtract(x,whitening='ZCA',nodes=2,sparsity=FALSE)
model = Backpropagation(feature[1:1000,],y[1:1000],nodes=10,
                        mission='classification',
                        alpha=0.1,lambda=0,maxStep=500)
W = model[[1]]
b = model[[2]]
fitted = t(ForwardPropagation(feature[1:1000,],W,b,'single'))
pred = t(ForwardPropagation(feature[1001:2000,],W,b,'single'))

auc(roc(fitted[,ncol(fitted)],as.factor(y[1:1000])))
auc(roc(pred[,ncol(pred)],as.factor(y[1001:2000])))

#Sparse Autoencoder
feature = FeatureExtract(x,whitening='ZCA',nodes=40,sparsity=TRUE)
model = Backpropagation(feature[1:1000,],y[1:1000],nodes=10,
                        mission='classification',
                        alpha=0.1,lambda=0,maxStep=2000)
W = model[[1]]
b = model[[2]]
fitted = t(ForwardPropagation(feature[1:1000,],W,b,'single'))
pred = t(ForwardPropagation(feature[1001:2000,],W,b,'single'))

auc(roc(fitted[,ncol(fitted)],as.factor(y[1:1000])))
auc(roc(pred[,ncol(pred)],as.factor(y[1001:2000])))
