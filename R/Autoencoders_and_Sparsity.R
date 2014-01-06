Autoencoder = function(x,nodes=5,alpha,lambda,maxStep)
{
    if (is.vector(x) || ncol(x)==1)
        stop('No need to encode!')
    model = Backpropagation(x=x,y=x,nodes=nodes,alpha=alpha,lambda=lambda,
                            maxStep=maxStep)
    model
}

SparseAutoencoder = function(x,nodes,alpha,beta,lambda,rho,maxStep)
{
    f = function(z) 1/(1+exp(-z))
    df = function(z) f(z)*(1-f(z))
    
    #Using stochastic gradient descent
    steps = 1
    m = nrow(x)
    ind = sample(1:m,maxStep,replace=T)
    if (is.vector(x) || ncol(x)==1)
        stop('No need to encode!')
    dx = x[ind,]
    
    tmp = ParameterInitializer(c(ncol(x),nodes,ncol(x)))
    W = tmp[[1]]
    b = tmp[[2]]
    
    stop_condition = FALSE
    cat('\r')
    while(!stop_condition)
    {
        if (steps%%100==0)
            cat(steps,'\r')
        tmp = ForwardPropagation(dx[steps,],W,b)
        a = tmp[[1]]
        z = tmp[[2]]
        n = length(a)
        
        #b2 = do.call(cbind,rep(list(b[[1]]),nrow(x)))
        a2 = W[[1]]%*%t(x)+as.vector(b[[1]])#b2#matrix(rep(b[[1]],nrow(x)),ncol=nrow(x))
        r = rowMeans(a2)
        r = beta*((1-rho)/(1-r)-rho/r)
        
        delta = a
        delta[[n]] = -(dx[steps,]-a[[n]])*df(z[[n]])
        #for (i in (n-1):2)
        i = 2
        delta[[i]] = (t(W[[i]])%*%delta[[i+1]]+r)*df(z[[i]])
        for (i in (n-1):1)
        {
            W[[i]] = W[[i]]-alpha*(delta[[i+1]]%*%t(a[[i]])+lambda*W[[i]])
            b[[i]] = b[[i]]-alpha*delta[[i+1]]
        }
        if (steps>=maxStep)
            stop_condition = TRUE
        steps = steps+1
    }
    list(W,b)
}