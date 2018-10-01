#' Reparameterised VB gradient ascent.
#'
#' This function runs a Variational Bayes gradient ascent to find the parameters of an approximating distribution that minimises the KL divergence to the true posterior.
#' Given a model function that calculates the derivative of the ELBO with resepct to lambda, it will return a list containing the optimal lambda,
#' history of ELBO values, and iterations required to converge for a given model function and dataset.
#' @param data Data passed directly into model, may also be used in conjunction with batch.
#' @param lambda Parameter vector to be optimised. If lambda is a matrix, each column is assumed to be the parameter vector for a component of a mixture distribution,
#'  with the first row corresponding to unnormalised weights z, where the normalised weights are given by p = exp(z) / sum(exp(z))
#' @param model A function with first three arguments: data, lambda, epsilon that returns a list with elements 'grad', the vector of the gradient of the ELBO with respect to lambda, and 'val', the value of the ELBO for this lambda.
#' @param dimTheta Integer. The dimension of the theta parameter vector
#' @param S Integer. The number of Monte Carlo estimates per iteration. Defaults to 25.
#' @param epsDist Either 'normal' or 'uniform'. Distribution of the auxiliary random variable. Defaults to 'normal'
#' @param batch Integer. If data is a matrix, calculate gradient based on batch many columns per iteration, if ncol(data) is divisible by batch. If batch = 0 use the whole data object per iteration.
#' @param maxIter Interger. Maximum number of gradient ascent iterations. Defaults to 5000.
#' @param alpha adam optimisation control parameter. Defaults to 0.01.
#' @param beta1 adam optimisation control parameter. Defaults to 0.9.
#' @param beta2 adam optimisation control parameter. Defaults to 0.99.
#' @param rollingWindowSize Integer. Take the mean of this many most recent iterations to assess convergence. Defaults to 5 if batch = 0, or ncol(data) / batch otherwise.
#' @param threshold Maximum difference in mean value of ELBO before convergence is achieved. Defaults to 0.01.
#' @param RQMC Boolean, if true simulate epsilon from Randomised Quasi Monte Carlo based off the sobol seqence. Defaults to FALSE.
#' @param ... Extra arguments passed into model
#' @export
reparamVB <- function(data, lambda, model, dimTheta, S = 25, epsDist = 'normal', batch = 0, maxIter = 5000, 
                       alpha = 0.01, beta1 = 0.9, beta2 = 0.99, rollingWindowSize = 5, threshold = 0.01, RQMC = FALSE,  ...){
  if(!is.matrix(lambda)){
    lambda <- matrix(lambda, ncol = 1)
  }
  mixtureComponents <- ncol(lambda)
  # Extract weights from lambda in a mixture distribution
  if(mixtureComponents > 1){
    zWeights <- lambda[1,]
    lambda <- lambda[2:nrow(lambda), ]
  }

  if(batch > 0 & is.matrix(data)){
    rollingWindowSize <- ncol(data) / batch
    if(!is.integer(rollingWindowSize)){
      stop('Number of columns of data must be divisible by batch')
    }
    subset <- c(0, seq(batch, ncol(data), batch))
    dataFull <- data
  } else if(batch > 0 & !is.matrix(data)){
    stop('batch > 0 is only available for matrix data')
  }
  if(!epsDist %in% c('normal', 'uniform')){
    stop('epsDist must be normal or uniform')
  }
  
  if(RQMC){
    genFile <- system.file("extdata", "sobolGenfile", package = "VBfuns", mustWork = TRUE)
    sobol <- sobolPoints(S, dimTheta * mixtureComponents, skip = 100, genFile)
  }

  diff <- threshold + 1
  iter <- 1
  LB <- numeric(maxIter)
  M <- V <- matrix(0, nrow(lambda), ncol(lambda))
  MZ <- VZ <- rep(0, mixtureComponents)
  e <- 1e-8
  meanLB <- 0
  oldMeanLB <- 0
  while(diff > threshold){
    if(iter > maxIter){
      break
    }
    if(batch > 0){
      set <- iter %% rollingWindowSize + 1
      data <- dataFull[,(subset[set]+1):subset[set+1]]
    }
    eval <- logq <- matrix(0, S, mixtureComponents)
    grad <- array(0, dim = c(nrow(lambda), S, mixtureComponents))

    if(RQMC){
      unif <- shuffleRQMC(sobol)
      unif[unif < 0.001] = 0.001
      unif[unif > 0.999] = 0.999
    } else {
      unif <- matrix(runif(S*dimTheta*mixtureComponents), nrow = S)
    }

    for(s in 1:S){
      for(m in 1:mixtureComponents){
        if(S == 1){
          epsilon <- unif[(m-1)*dimTheta + 1:dimTheta]
        } else {
          epsilon <- unif[s, (m-1)*dimTheta + 1:dimTheta]
        }
        logq[s, m] <- 0
        if(epsDist == 'normal'){
          epsilon <- qnorm(epsilon)
          logq[s, m] <- sum(dnorm(epsilon, log = TRUE))
        }
        logpj <- model(data,
                       as.matrix(lambda[, m]),
                       epsilon,
                       ...)
        eval[s, m] <- logpj$val
        grad[,s, m] <- logpj$grad
      }
    }
    # Mean of gradient
    gradient <- apply(grad, c(1, 3), mean, na.rm = TRUE)
    gradientSq <- gradient^2
    eval[is.infinite(eval)] <- NA
    LB[iter] <- mean(eval - logq, na.rm=TRUE)
    # Multiply gradients by the relevant mixture weight, and calculate the gradient of z
    if(mixtureComponents > 1){
      # Normalise
      pWeights <- exp(zWeights) / sum(exp(zWeights))
      # Weights gradients by the normalised weigths vector
      gradient <- t(t(gradient * pWeights))
      gradientSq <- gradient^2
      gradZ <- rep(0, mixtureComponents)
      # Gradient of z uses the sum of dELBO / dp * dp /dz over each p
      denom <- sum(exp(zWeights))^2
      q <- exp(logq)
      qRatio <- t(apply(q, 1, function(x) x / sum(x)))

      for(i in 1:mixtureComponents){
        for(j in 1:mixtureComponents){
          if(i == j){
            gradZ[i] <- gradZ[i] + mean(eval[,j] - logq[,j] - qRatio[,j], na.rm = TRUE) * sum(exp(zWeights[i] + zWeights[-i])) / denom
          } else {
            gradZ[i] <- gradZ[i] - mean(eval[,j] - logq[,j] - qRatio[,j], na.rm = TRUE) * exp(zWeights[i] + zWeights[j]) / denom
          }
        }
      }
      gradZSq <- gradZ^2
      # ELBO = sum p_i L_i
      LB[iter] <- colMeans(eval - logq, na.rm = TRUE) %*% pWeights
    }

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

    if(mixtureComponents > 1){
      MZ <- beta1 * MZ + (1 - beta1) * gradZ
      VZ <- beta2 * VZ + (1 - beta2) * gradZSq
      MstarZ <- MZ / (1 - beta1^iter)
      VstarZ <- VZ / (1 - beta2^iter)
      update <- alpha * MstarZ / (sqrt(VstarZ) + e)
      if(any(is.na(update))){
        print('Break')
        break
      }
      zWeights <- zWeights + update
    }

    if(iter %% rollingWindowSize == 0){
      oldMeanLB <- meanLB
      meanLB <- mean(LB[iter:(iter-(rollingWindowSize - 1))])
      diff <- abs(meanLB - oldMeanLB)
    }
    if(iter %% 100 == 0){
        print(paste0('Iteration: ', iter, ' ELBO: ', meanLB))
    }
    iter <- iter + 1
  }
  print(paste0('iter: ', min(iter-1, maxIter), ' ELBO: ', meanLB))
  if(mixtureComponents == 1){
    return(list(lambda=lambda, LB = LB[1:min(iter-1, maxIter)], iter = min(maxIter, iter-1)))
  } else {
    return(list(lambda=lambda, weights = zWeights, LB = LB[1:min(iter-1, maxIter)], iter = min(maxIter, iter-1)))
  }
}
