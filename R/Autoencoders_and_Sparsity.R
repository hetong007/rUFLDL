Autoencoder = function(x,nodes=5,alpha,lambda,maxStep)
{
    if (is.vector(x) || ncol(x)==1)
        stop('No need to encode!')
    model = Backpropagation(x=x,y=x,nodes=nodes,mission='regression',
                            alpha=alpha,lambda=lambda,maxStep=maxStep)
    model
}

SparseAutoencoder = function(x,nodes,alpha,beta,lambda,rho,maxStep)
{
    f = function(z) as.vector(1/(1+exp(-z)))
    df = function(z) as.vector(f(z)*(1-f(z)))
    
    #Using batch gradient descent
    
    m = nrow(x)
    y = x
    #ind = sample(1:m,maxStep,replace=T)

    mny = min(y)
    mxy = max(y)
    if (mny>=0 && mxy<=1)
    {
        mny = 0
        mxy = 1
    }
    dy = t(x)
    
    tmp = ParameterInitializer(c(ncol(x),nodes,nrow(dy)))
    W = tmp[[1]]
    b = tmp[[2]]
    
    stop_condition = FALSE
    steps = 1
    cat('\r')
    while(!stop_condition)
    {
        if (steps%%100==0)
            cat(steps,'\r')
        tmp = ForwardPropagation(x,W,b)
        a = tmp[[1]]
        z = tmp[[2]]
        n = length(a)
        
        delta = a
        r = W[[1]]%*%t(x)+as.vector(b[[1]])
        r = beta*((1-rho)/(1-r)-rho/r)/m
        
        delta[[n]] = -((dy-mny)/(mxy-mny)-a[[n]])*df(z[[n]])
        for (i in (n-1):2)
            delta[[i]] = (t(W[[i]])%*%delta[[i+1]]+r)*df(z[[i]])
        for (i in (n-1):1)
        {
            W[[i]] = W[[i]]-alpha*(delta[[i+1]]%*%t(a[[i]])/m+lambda*W[[i]])
            b[[i]] = b[[i]]-alpha*delta[[i+1]]/m
        }
        
        if (steps>=maxStep)
            stop_condition = TRUE
        steps = steps+1
    }
    list(W,b)
}
