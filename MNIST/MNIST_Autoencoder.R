load('MNIST/data/train.rda')
x = as.matrix(train[1:2000,-1])
source('MNIST/Plot.R')
Plot(x,5,1:25)

#Autoencoder with small number of hidden layers
model = Autoencoder(x[1:25,],nodes=500,alpha=0.1,lambda=0.0001,maxStep=500)
W = model[[1]]
b = model[[2]]
fitted = ForwardPropagation(x[1:25,],W,b,'single')
mnx = min(x)
mxx = max(x)
fitted = fitted*(mxx-mnx)+mnx
fitted = t(fitted)
Plot(x[1:25,]-fitted[1:25,],5,1:25)
hist(x[1:25,]-fitted[1:25,])

#Autoencoder with sparsity penalty of hidden layers
model = SparseAutoencoder(x[1:25,],nodes=500,corruption_level=0.2,
                          alpha=0.1,beta=3,lambda=3e-3,rho=0.2,maxStep=50)
W = model[[1]]
b = model[[2]]
fitted = ForwardPropagation(x[1:25,],W,b,'single')
mnx = min(x)
mxx = max(x)
fitted = fitted*(mxx-mnx)+mnx
fitted = t(fitted)
Plot(x[1:25,]-fitted,5,1:25)
hist(x[1:25,]-fitted[1:25,])
