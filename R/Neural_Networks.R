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

ForwardPropagation = function(x,W=NULL,b=NULL,output='all',last='logistic')
{
    f = function(z) 1/(1+exp(-z))
    if (is.null(W) || is.null(b))
        stop('Not enough input.')
    n = length(W)
    if (length(W)==length(b)+1 && last == 'logistic')
    {
        last = 'softmax'
        warning('The last layer changed to softmax output')
    }
    if (!is.matrix(x))
        x = matrix(x)
    else
        x = t(x)
    a = vector(n+1,mode='list')
    a[[1]] = x
    z = a
    if (last=='logistic')
    {
        for (i in 1:n)
        {
            z[[i+1]] = W[[i]]%*%a[[i]]+as.vector(b[[i]])
            a[[i+1]] = f(z[[i+1]])
        }
    }
    else if (last=='softmax')
    {
        for (i in 1:(n-1))
        {
            z[[i+1]] = W[[i]]%*%a[[i]]+as.vector(b[[i]])
            a[[i+1]] = f(z[[i+1]])
        }
        i=n
        z[[i+1]] = W[[n]]%*%a[[n]]
        a[[i+1]] = exp(z[[i+1]])
        cs = Diagonal(x=1/colSums(a[[i+1]]))
        a[[i+1]] = as.matrix(a[[i+1]] %*% cs)
    }
    else if (last=='linear')
    {
        for (i in 1:(n-1))
        {
            z[[i+1]] = W[[i]]%*%a[[i]]+as.vector(b[[i]])
            a[[i+1]] = f(z[[i+1]])
        }
        i=n
        z[[i+1]] = W[[i]]%*%a[[i]]+as.vector(b[[i]])
        a[[i+1]] = z[[i+1]]
    }
    else
        stop('Invalid last')
    if (output=='all')
        return(list(a,z))
    if (output=='encoder')
        return(a[[2]])
    if (output=='single')
        return(a[[n+1]])
}

