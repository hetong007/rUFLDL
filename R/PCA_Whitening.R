Whitening = function(x,contribution=0.9,type='PCA',eps=1e-5)
{
    pca = princomp(x)
    cont = cumsum(pca$sdev^2)/sum(pca$sdev^2)
    k = min(sum(cont<=contribution)+1,ncol(x))
    
    lambda = diag(1/sqrt(pca$sdev+eps))
    ans = pca$scores %*% lambda
    if (type=='PCA')
        return(ans[,1:k])
    ans = t(as(pca$loadings,'matrix') %*% t(ans))
    ans
}
