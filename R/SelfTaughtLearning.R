FeatureExtract = function(x,whitening=NULL,nodes,sparsity=FALSE,
                          maxStep=2000,output='feature')
{
    if (is.vector(x) || ncol(x)==1)
        stop('No need to encode!')
    if (!is.null(whitening))
    {
        if (whitening=='PCA')
            x = Whitening(x,type='PCA')
        else if (whitening=='ZCA')
            x = Whitening(x,type='ZCA')
    }
    #if (nodes<ncol(x) && sparsity)
        #stop('No need to use sparse Autoencoder, 
             #please check your layer setting.')
    
    if (sparsity)
        model = SparseAutoencoder(x,nodes,alpha=0.1,beta=0.1,lambda=0,rho=0.1,
                                  maxStep=maxStep)
    else
        model = Autoencoder(x,nodes,alpha=0.1,lambda=0,maxStep=maxStep)
    
    W = model[[1]]
    b = model[[2]]
    
    if (output=='feature')
    {
        feature = ForwardPropagation(x,W,b,output='encoder')
        return(t(feature))
    }
    return(list(W,b))
}
