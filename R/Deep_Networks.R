DeepNetworks = function(x,y,nodes_1=40,maxStep_1=100,nodes_2=3,maxStep_2=1000,
                        alpha=0.1,lambda=0.001,maxStep_back = 10000)
{
    para_1 = FeatureExtract(x,whitening='ZCA',nodes=nodes_1,
                            sparsity=TRUE,maxStep=maxStep_1,
                            output='parameters')
    W = para_1[[1]]
    b = para_1[[2]]
    feature_1 = t(apply(x,1,ForwardPropagation,W,b,output='encoder'))
    
    para_2 = FeatureExtract(feature_1,whitening=NULL,nodes=3,
                            sparsity=FALSE,maxStep=maxStep_2,
                            output='parameters')
    
    if (is.vector(x))
        x = as.matrix(x)
    if (is.vector(y))
        y = as.matrix(y)
    tmp=ParameterInitializer(c(ncol(x),nodes_1,nodes_2,ncol(y)))
    W = tmp[[1]]
    b = tmp[[2]]
    W[[1]] = para_1[[1]][[1]]
    W[[2]] = para_2[[1]][[1]]
    b[[1]] = para_1[[2]][[1]]
    b[[2]] = para_2[[2]][[1]]
    
    model = Backpropagation(x=x,y=y,nodes=c(nodes_1,nodes_2),W=W,b=b,
                            alpha=alpha,lambda=lambda,maxStep=maxStep_back)
    return(model)
}
