x = matrix(runif(6000),ncol=6)
x[,1] = runif(1000,min=0,max=0.1)
x[,2] = x[,1]+5
x[,3] = x[,2]*5
x[,4] = runif(1000,min=0.9,max=1)
x[,5] = x[,4]+5
x[,6] = x[,5]*5
for (i in 1:6)
    x[,i] = (x[,i]-min(x[,i]))/(max(x[,i])-min(x[,i]))


#Autoencoder with small number of hidden layers
model = Autoencoder(x,nodes=2,alpha=0.1,lambda=0,maxStep=10000)
W = model[[1]]
b = model[[2]]
fitted = apply(x,1,ForwardPropagation,W,b,'single')
fitted = t(fitted)
plot(fitted,x,pch=20)

#Autoencoder with sparsity penalty of hidden layers
model = SparseAutoencoder(x,nodes=20,alpha=0.1,beta=0.1,lambda=0,rho=0.1,
                          maxStep=10000)
W = model[[1]]
b = model[[2]]
fitted = apply(x,1,ForwardPropagation,W,b,'single')
fitted = t(fitted)
plot(fitted,x,pch=20)
