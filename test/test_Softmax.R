x = matrix(rnorm(10000),ncol=5)

y = x[,1]-x[,2]*x[,4]+x[,3]^2
y = findInterval(y,quantile(y,c(0.3,0.6)))


model = Softmax(x[1:1000,],y[1:1000],0.1,0.01)

fitted = model[[2]]
pred = apply(model[[4]] %*% t(x[1001:2000,]),2,max)

require(AUC)
auc(roc(fitted,as.factor(y[1:1000])))
auc(roc(pred,as.factor(y[1001:2000])))
