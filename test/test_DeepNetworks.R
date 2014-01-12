x = matrix(rnorm(10000),ncol=5)

g = function(x) 1/(1+2^(x[1]^3+x[2]+x[1]*x[2]))
y = as.numeric(apply(x[,1:2],1,g)>runif(2000))

model = DeepNetworks(x[1:1000,],y[1:1000])
W = model[[1]]
b = model[[2]]
fitted = t(ForwardPropagation(x[1:1000,],W,b,'single'))
pred = t(ForwardPropagation(x[1001:2000,],W,b,'single'))

auc(roc(fitted[,ncol(fitted)],as.factor(y[1:1000])))
auc(roc(pred[,ncol(pred)],as.factor(y[1001:2000])))