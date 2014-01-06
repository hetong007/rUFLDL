Softmax = function(x,y,alpha,lambda,eps=1e-5,maxStep=1000)
{
    n = nrow(x)
    p = ncol(x)
    k = length(unique(y))
    theta = matrix(rnorm(p*k),ncol=p)
    
    stop_condition = FALSE
    steps = 1
    cat('\r')
    
    while(!stop_condition)
    {
        if (steps%%10==0)
           cat(steps,sum((old_theta-theta)^2),'\r')
        old_theta = theta
        Mat = exp(theta %*% t(x))
        cs = colSums(Mat)
        for (j in 1:k)
        {
            id = as.numeric(y==j)
            #dg = diag(id-Mat[j,]/cs)
            dg = Diagonal(x=id-Mat[j,]/cs)
            delta = -Matrix::rowMeans(x=t(x)%*%dg)
            theta[j,] = (1-alpha*lambda)*theta[j,]-alpha*delta
        }
        if (steps>=maxStep)
            stop_condition = TRUE
        if (sum((old_theta-theta)^2)<eps)
            stop_condition = TRUE
        steps = steps+1 
    }
    pred = t(theta %*% t(x))
    ans = apply(pred,1,which.max)
    prob = apply(pred,1,max)
    list(pred,prob,ans,theta)
}