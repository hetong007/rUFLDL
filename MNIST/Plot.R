Plot = function(data,ncol,ind)
{
    #A matrix matrix(as.matrix(data[ind,-1]),ncol=28)
    nrow = length(ind)/ncol
    if (nrow*ncol!=length(ind))
        stop('nrow*ncol!=length(ind)')
    img = mat.or.vec(nrow*28,ncol*28)
    for (i in 1:nrow)
        for (j in 1:ncol)
        {
            rowind = (28*(i-1)+1):(28*i)
            colind = (28*(j-1)+1):(28*j)
            tind = ind[(i-1)*ncol+j]
            img[rowind,colind] = matrix(as.matrix(data[tind,]),ncol=28)
        }
    image(img[,ncol(img):1], col = grey(seq(0, 1, length = 256)))
}