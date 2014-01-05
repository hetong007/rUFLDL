x = matrix(rnorm(1000),ncol=5)
y = x[,1]+x[,2]-x[,3]+sin(x[,4])
y = as.numeric(y>0)

tmp=ParameterInitializer(c(5,10,1))
W = tmp[[1]]
b = tmp[[2]]

model = Backpropagation(x,y,W,b,alpha=0.1,lambda=1,maxStep=100000)
W = model[[1]]
b = model[[2]]
pred = apply(x,1,ForwardPropagation,W,b,'single')

require(AUC)
auc(roc(pred,as.factor(y)))
