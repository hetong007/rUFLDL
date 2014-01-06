FeatureExtract = function(x,whitening=NULL,layers,sparsity=FALSE)
{
    if (!is.null(whitening))
    {
        if (whitening=='PCA')
            x = Whitening(x,type='PCA')
        else if (whitening=='ZCA')
            x = Whitening(x,type='ZCA')
    }
    if (layers[2]<layers[1] && sparsity)
        stop('No need to use sparse Autoencoder, please check your layer setting.')
    Init=ParameterInitializer(layers)
    W = Init[[1]]
    b = Init[[2]]
    
    if (sparsity)
        model = SparseAutoencoder(x,W,b,
                                  alpha=0.1,beta=0.1,lambda=0,rho=0.1,
                                  maxStep=10000)
    else
        model = Autoencoder(x,W,b,alpha=0.1,lambda=0,maxStep=10000)
    
    W = model[[1]]
    b = model[[2]]
    
    feature = apply(x,1,ForwardPropagation,W,b,output='encoder')
    t(feature)
}
