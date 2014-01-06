ParameterInitializer = function(layers)
{
    n = length(layers)
    W = vector(n-1,mode='list')
    b = vector(n-1,mode='list')
    for (i in 1:(n-1))
    {
        W[[i]] = matrix(rnorm(layers[i]*layers[i+1],mean=0,sd=0.1),
                        ncol=layers[i])
        b[[i]] = rnorm(layers[i+1],mean=0,sd=0.1)
    }
    list(W,b)
}

ForwardPropagation = function(x,W=NULL,b=NULL,output='all')
{
    f = function(z) 1/(1+exp(-z))
    if (is.null(W) || is.null(b))
        stop('Not enough input.')
    n = length(W)
    if (n!=length(b))
        stop('Parameters\' lengths differs')
    a = vector(n+1,mode='list')
    a[[1]] = x
    z = a
    for (i in 1:n)
    {
        z[[i+1]] = W[[i]]%*%a[[i]]+b[[i]]
        a[[i+1]] = f(z[[i+1]])
    }
    if (output=='all')
        return(list(a,z))
    if (output=='encoder')
        return(a[[2]])
    if (output=='single')
        return(a[[n+1]])
}

