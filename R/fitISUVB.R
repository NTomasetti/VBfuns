#' Generic function for IS-UVB gradient ascent
#'
#' Returns the vector of optimal lambda values, the history of ELBO values, the number of iterations to converge, and the history of the effective sample size
#' @param lambda The vector or one column matrix to be optimised
#' @param qDist A function with first two arguments: samples, lambda, that returns a list with two elements: grad, the gradient of log Q, and val, the value of log Q.
#' @param samples The matrix of samples from the previous UVB distribution. Each row must be a single draw from a multivariate distribution
#' @param dSamples A vector of the density of each row of samples according the the previous UVB distribution
#' @param logjoint A vector of the log of the joint density between the data and each row of samples
#' @param maxIter Interger. Maximum number of gradient ascent iterations.
#' @param alpha adam optimisation control parameter.
#' @param beta1 adam optimisation control parameter.
#' @param beta2 adam optimisation control parameter.
#' @param rollingWindowSize Integer. Take the mean of this many most recent iterations to assess convergence. Defaults to 5
#' @param threshold Maximum difference in mean value of ELBO before convergence is achieved.
#' @param ... Extra arguments passed to qDist
#' @export
fitISUVB <- function(lambda, qDist, samples, dSamples, logjoint, maxIter = 5000, alpha = 0.01, beta1 = 0.9, beta2 = 0.99, threshold = 0.01, ...){
  if(!is.matrix(lambda)){
    lambda <- matrix(lambda, ncol = 1)
  }
  dimLambda <- nrow(lambda)
  if(!is.matrix(samples)){
    # samples may be passed in as a vector if theta is one-dimensional
    samples <- matrix(samples, ncol = 1)
  }
  diff <- threshold + 1
  iter <- 1
  maxIter <- 2000
  LB <- ESS <- numeric(maxIter)
  M <- V <- numeric(dimLambda)
  e <- 1e-8
  meanLB <- 0
  oldMeanLB <- 0
  S <- nrow(samples)
  while(diff > threshold){
    if(iter > maxIter){
      break
    }

    eval <- numeric(S)
    dLq <- matrix(0, S, dimLambda)
    lQ <- rep(0, S)
    for(s in 1:S){
      model <- qDist(samples[s,], lambda, ...)
      lQ[s] <- model$val
      dLq[s,] <- model$grad
    }

    w <- exp(lQ) / dSamples
    eval <- w * (logjoint - lQ)
    grad <- w * dLq * (logjoint - lQ)
    f <- w * dLq

    a <-  diag(cov(grad, f) / var(f))
    a[is.na(a)] <- 0

    gradient <- colMeans(grad - a * f)
    gradientSq <- colMeans((grad - a * f)^2)
    LB[iter] <- mean(eval)

    M <- beta1 * M + (1 - beta1) * gradient
    V <- beta2 * V + (1 - beta2) * gradientSq
    Mstar <- M / (1 - beta1^iter)
    Vstar <- V / (1 - beta2^iter)
    update <- alpha * Mstar / (sqrt(Vstar) + e)
    if(any(is.na(update))){
      print('Break')
      break
    }
    lambda <- lambda + update

    wNorm <- w / sum(w)
    ESS[iter] <- 1 / sum(wNorm^2)

    if(iter %% rollingWindowSize == 0){
      oldMeanLB <- meanLB
      meanLB <- mean(LB[iter:(iter-rollingWindowSize)])
      diff <- abs(meanLB - oldMeanLB)
    }
    if(iter %% 100 == 0){
       print(paste0('Iteration: ', iter, ' ELBO: ', meanLB))
    }
    iter <- iter + 1
  }
  print(paste0('iter: ', min(iter-1, maxIter), ' ELBO: ', meanLB))
  return(list(lambda=lambda,
              LB = LB[1:min(iter-1, maxIter)],
              iter = min(maxIter, iter-1),
              ESS = ESS[1:min(iter-1, maxIter)]))
}
