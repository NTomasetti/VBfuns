#' Stein VB to optimise a particle cloud for a given model
#'
#' This function inputs a particle cloud and derivative function, and returns the KL divergence minimising particle cloud.
#' @param particles An N x P matrix of N starting values of a P dimensional theta
#' @param model A function that inputs theta as its first argument that returns a list with two values, grad, the P dimensional vector of the derivative of the log joint density with respect to theta, and val, the value of the log joint density at this value
#' @param kernel A kernel function that inputs a vector x1, a vector x2, and smoothing parameter(s) h and returns a list with two values, grad, the P dimensional vector of the derivative of the kernel  with respect to X1, and val, the value of the kernel at these inputs.
#' Defaults to RBFkernel, which has code provided in this package.
#' @param Nsub The size of the subset of the particles to evaluate the gradient over, defaults to N
#' @param h kernel smoothing parameter, defaults to 0.01
#' @param alpha learning rate, defaults to 0.001
#' @param maxIter maximum number of iterations, defaults to 1000
#' @param rollingWindowSize Integer. Take the mean of this many most recent iterations to assess convergence. Defaults to 5.
#' @param threshold Maximum difference in mean value of ELBO before convergence is achieved. Defaults to 0.01.
#' @param ... Extra arguments to pass to model
#' @export
steinVB <- function(particles, model, kernel = RBFkernel, Nsub = nrow(particles), h = 0.01, alpha = 1e-3, maxIter = 1000, rollingWindowSize = 5, threshold = 1e-3, ...){
  iter <- 1
  LB <- rep(0, maxIter)
  diff <- threshold + 1
  M <- V <- rep(0, Nsub)
  e <- 1e-8
  oldLB <- 0

  while(diff > threshold){
    if(iter > maxIter){
      break
    }

    phiHat <- matrix(0, nrow(particles), ncol(particles))
    logP <- matrix(0, Nsub, ncol(particles))
    elbo <- rep(0, Nsub)

    for(i in 1:Nsub){
      stein <- model(particles[i, ], ...)
      elbo[i] <- stein$val
      logP[i, ] <- stein$grad
    }

    for(i in 1:ncol(particles)){
      for(j in 1:Nsub){
        kernel <- kernel(particles[i, ], as.matrix(particles[j, ]), h)
        phiHat[i, ] <- phiHat[i, ] + 1/N * (kernel$val * logP[j, ] + kernel$grad)
      }
    }
    LB[iter] <- mean(elbo)

    particles <- particles + alpha * phiHat


    if(iter %% rollingWindowSize == 0){
      meanLB <- mean(LB[iter:(iter- rollingWindowSize+1)])
      diff <- abs(meanLB - oldLB)
      oldLB <- meanLB
    }
    if(iter %% 25 == 0){
      print(paste0('Iteration: ', iter, ', ELBO: ', meanLB))
    }

    iter <- iter + 1
  }
  print(paste0('Converged at Iteration: ', iter - 1, ' at ELBO: ', meanLB))
  particles
}
