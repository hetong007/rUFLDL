Softmax = function(x,y,alpha,lambda,eps=0,maxStep=1000)
{
    if (is.vector(x))
        x = as.matrix(x)
    x = cbind(1,x)
    m = nrow(x)
    p = ncol(x)
    k = length(unique(y))
    dy = NULL
    for (i in 0:(k-1))
        dy = cbind(dy,as.numeric(y==i))
    theta = matrix(rnorm(p*k,sd=0.01),ncol=p)
    #ind = sample(1:m,maxStep,replace=T)
    #dx = x[ind,,drop=FALSE]
    #dy = dy[ind,,drop=FALSE]
    
    stop_condition = FALSE
    steps = 1
    cat('\r')
    
    while(!stop_condition)
    {
        if (steps%%10==0)
           cat(steps,sum((old_theta-theta)^2),'\r')#,cost,'\r')
        old_theta = theta
        #Mat = exp(theta %*% dx[steps,])
        #grad = -(dy[steps,]-Mat/sum(Mat)) %*% dx[steps,]
        
        Mat = exp(theta %*% t(x))
        cs = Diagonal(x=1/colSums(Mat))
        Mat = as.matrix(Mat %*% cs)
        grad = lambda*theta-(t(dy)-Mat)%*%x/m
        
        #cost = -(1/m)*sum(t(dy)*log(Mat))+lambda/2*sum(theta^2)
        
        theta = theta-alpha*grad
        
        if (steps>=maxStep)
            stop_condition = TRUE
        if (sum((old_theta-theta)^2)<eps)
            stop_condition = TRUE
        steps = steps+1 
    }
    #Mat = exp(theta %*% t(x))
    #cs = Diagonal(x=1/colSums(Mat))
    #pred = t(as(Mat %*% cs,'matrix'))
    pred = as.matrix(t(theta %*% t(x)))
    prob = apply(pred,1,max)
    ans = apply(pred,1,which.max)-1
    list(pred,prob,ans,theta)
}
