Backpropagation = function(x,y,nodes=5,W=NULL,b=NULL,
                           mission='classification',
                           alpha,lambda,maxStep,linearity = FALSE)
{
    f = function(z) as.vector(1/(1+exp(-z)))
    df = function(z) as.vector(f(z)*(1-f(z)))
    
    #Using batch gradient descent
        
    m = nrow(x)
    if (mission=='classification')
    {
        k = length(unique(y))
        dy = NULL
        for (i in 0:(k-1))
            dy = cbind(dy,as.numeric(y==i))
        dy = t(dy)
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
        if (is.vector(y))
            dy = t(matrix(y))
        else
            dy = t(y)
    }
    else
        stop('Invalid mission.')
    
    if (is.null(W) || is.null(b))
    {
        tmp = ParameterInitializer(c(ncol(x),nodes,nrow(dy)))
        W = tmp[[1]]
        b = tmp[[2]]
    }
    
    stop_condition = FALSE
    steps = 1
    cat('\r')
    while(!stop_condition)
    {
        cat(steps,'\r')
        #tmp = ForwardPropagation(dx[steps,],W,b)
        if (linearity)
            tmp = ForwardPropagation(x,W,b,last='linear')
        else
            tmp = ForwardPropagation(x,W,b)
        a = tmp[[1]]
        z = tmp[[2]]
        n = length(a)
        
        delta = a

        if (linearity)
            delta[[n]] = -(dy-a[[n]])
        else if (mission=='classification')
            delta[[n]] = -(dy-a[[n]])*df(z[[n]])
        else
            delta[[n]] = -((dy-mny)/(mxy-mny)-a[[n]])*df(z[[n]])
        
        for (i in (n-1):2)
            delta[[i]] = (t(W[[i]])%*%delta[[i+1]])*df(z[[i]])
        for (i in (n-1):1)
        {
            W[[i]] = W[[i]]-alpha*(delta[[i+1]]%*%t(a[[i]])/m+lambda*W[[i]])
            b[[i]] = b[[i]]-rowMeans(alpha*delta[[i+1]])
        }

        if (steps>=maxStep)
            stop_condition = TRUE
        steps = steps+1
    }
    list(W,b)
}
