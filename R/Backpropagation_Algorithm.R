Backpropagation = function(x,y,nodes=5,mission='classification',W=NULL,b=NULL,alpha,lambda
                           ,maxStep)
{
    f = function(z) as.vector(1/(1+exp(-z)))
    df = function(z) as.vector(f(z)*(1-f(z)))
    
    #Using stochastic gradient descent
    steps = 1
    m = nrow(x)
    ind = sample(1:m,maxStep,replace=T)
    if (is.vector(x))
        x = as.matrix(x)
    if (mission=='classification')
    {
        k = length(unique(y))
        dy = NULL
        for (i in 0:(k-1))
            dy = cbind(dy,as.numeric(y==i))
    }
    else if (mission=='regression')
    {
        mny = min(y)
        mxy = max(y)
        if (mny>=0 && mxy<=1)
        {
            mny = 0
            mxy = 1
        }
        dy = as.matrix(y)
    }
    else
        stop('Invalid mission.')
    dx = x[ind,,drop=FALSE]
    dy = dy[ind,,drop=FALSE]
    
    if (is.null(W) || is.null(b))
    {
        tmp = ParameterInitializer(c(ncol(x),nodes,ncol(dy)))
        W = tmp[[1]]
        b = tmp[[2]]
    }
    
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
        
        delta = a

        if (mission=='classification')
            delta[[n]] = -(dy[steps,]-a[[n]])*df(z[[n]])
        else
            delta[[n]] = -((dy[steps,]-mny)/(mxy-mny)-a[[n]])*df(z[[n]])
        for (i in (n-1):2)
            delta[[i]] = (t(W[[i]])%*%delta[[i+1]])*df(z[[i]])
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
