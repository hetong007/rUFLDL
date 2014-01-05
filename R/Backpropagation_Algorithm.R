Backpropagation = function(x,y,W,b,alpha,lambda,maxStep)
{
    f = function(z) 1/(1+exp(-z))
    df = function(z) f(z)*(1-f(z))
    
    #Using stochastic gradient descent
    steps = 1
    m = nrow(x)
    ind = sample(1:m,maxStep,replace=T)
    dx = x[ind,]
    dy = y[ind]
    
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
        delta[[n]] = -(dy[steps]-a[[n]])*df(z[[n]])
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