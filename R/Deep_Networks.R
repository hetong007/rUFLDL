DeepNetworks = function(x,y,...)
{
    para_1 = FeatureExtract(x,whitening='ZCA',layers=c(6,40,6),
                            sparsity=TRUE,maxStep=100,output='parameters')
    W = para_1[[1]]
    b = para_1[[2]]
    feature_1 = t(apply(x,1,ForwardPropagation,W,b,output='encoder'))
    
    para_2 = FeatureExtract(feature_1,whitening=NULL,layers=c(40,3,40),
                           sparsity=FALSE,maxStep=1000,output='parameters')
    W = para_2[[1]]
    b = para_2[[2]]
    
    tmp=ParameterInitializer(c(6,40,3,1))
    W = tmp[[1]]
    b = tmp[[2]]
    W[[1]] = para_1[[1]][[1]]
    W[[2]] = para_2[[1]][[1]]
    b[[1]] = para_1[[2]][[1]]
    b[[2]] = para_2[[2]][[1]]
    
    model = Backpropagation(x,y,W,b,alpha=0.1,lambda=0,maxStep=10000)
    return(model)
}
