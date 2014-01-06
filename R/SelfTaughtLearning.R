FeatureExtract = function(x,whitening=NULL,layers,sparsity=FALSE,maxStep=10000,output='feature')
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
                                  maxStep=maxStep)
    else
        model = Autoencoder(x,W,b,alpha=0.1,lambda=0,maxStep=maxStep)
    
    W = model[[1]]
    b = model[[2]]
    
    if (output=='feature')
    {
        feature = apply(x,1,ForwardPropagation,W,b,output='encoder')
        return(t(feature))
    }
    return(list(W,b))
}
