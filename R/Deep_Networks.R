DeepNetworks = function(x,y,nodes_1=40,maxStep_1=100,sparsity_1=TRUE,
                        nodes_2=3,maxStep_2=100,sparsity_2=FALSE,
                        maxStep_3=1000,
                        alpha=0.1,lambda=0.001,maxStep_back=300)
{
    para_1 = FeatureExtract(x,whitening='ZCA',nodes=nodes_1,
                            sparsity=sparsity_1,maxStep=maxStep_1,
                            output='parameters')
    cat('Layer One finished!\n')
    W = para_1[[1]]
    b = para_1[[2]]
    feature_1 = t(ForwardPropagation(x,W,b,output='encoder'))
    
    para_2 = FeatureExtract(feature_1,whitening=NULL,nodes=nodes_2,
                            sparsity=sparsity_2,maxStep=maxStep_2,
                            output='parameters')
    cat('Layer Two finished!\n')
    W = para_2[[1]]
    b = para_2[[2]]
    feature_2 = t(ForwardPropagation(feature_1,W,b,output='encoder'))
    
    para_3 = Softmax(feature_2,y,alpha=alpha,lambda=lambda,maxStep=maxStep_3)
    para_3 = para_3[[4]]
    
    cat('Softmax finished!\n')
    if (is.vector(x))
        x = as.matrix(x)
    #if (is.vector(y))
        #y = as.matrix(y)
    k = length(unique(y))
    dy = NULL
    for (i in 0:(k-1))
        dy = cbind(dy,as.numeric(y==i))
    tmp=ParameterInitializer(c(ncol(x),nodes_1,nodes_2,ncol(dy)))
    W = tmp[[1]]
    b = tmp[[2]]
    W[[1]] = para_1[[1]][[1]]
    W[[2]] = para_2[[1]][[1]]
    W[[3]] = para_3[,-1]
    b[[1]] = para_1[[2]][[1]]
    b[[2]] = para_2[[2]][[1]]
    b[[3]] = para_3[,1]
    
    fitted = t(ForwardPropagation(x,W,b,'single'))
    cat(auc(roc(fitted[,ncol(fitted)],as.factor(y))),'\n')
    
    #Fine Tune
    #before softmax, precision is over 91%, after, 15% is reached ..... why?
    model = Backpropagation(x=x,y=y,nodes=c(nodes_1,nodes_2),
                            mission='classification',
                            W=W,b=b,alpha=alpha,lambda=lambda,
                            maxStep=maxStep_back)
    return(model)
}
