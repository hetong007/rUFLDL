x = matrix(rnorm(6000),ncol=6)
x[,1] = runif(1000,min=0,max=0.1)
x[,2] = x[,1]+5
x[,3] = x[,2]*5
x[,4] = runif(1000,min=0.9,max=1)
x[,5] = x[,4]+5
x[,6] = x[,5]*5

for (i in 1:6)
    x[,i] = (x[,i]-min(x[,i]))/(max(x[,i])-min(x[,i]))


tmp=ParameterInitializer(c(6,2,6))
W = tmp[[1]]
b = tmp[[2]]

model = Autoencoder(x,W,b,alpha=0.1,lambda=0,maxStep=10000)
W = model[[1]]
b = model[[2]]
fitted = apply(x,1,ForwardPropagation,W,b,'single')
fitted = t(fitted)
plot(fitted,x)
