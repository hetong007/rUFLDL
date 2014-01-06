x = matrix(rnorm(10000),ncol=5)

choice = runif(1)
show(choice)
if (choice>0.5) {
    y = x[,1]-exp(x[,2]^2)+sin(x[,4])
    y = as.numeric(y>0)
} else {
    g = function(x) 1/(1+2^(x[1]^3+x[2]+x[1]*x[2]))
    y = as.numeric(apply(x[,1:2],1,g)>runif(2000))
}

model = Backpropagation(x[1:1000,],y[1:1000],nodes=100,
                        alpha=0.1,lambda=0,maxStep=10000)
W = model[[1]]
b = model[[2]]
fitted = apply(x[1:1000,],1,ForwardPropagation,W,b,'single')
pred = apply(x[1001:2000,],1,ForwardPropagation,W,b,'single')

auc(roc(fitted,as.factor(y[1:1000])))
auc(roc(pred,as.factor(y[1001:2000])))

