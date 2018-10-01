// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

#include <RcppArmadillo.h>
#include <Rcpp.h>

//' Shuffles a low discrepancy sequence in the unit hypercube from Randomised Quasi-Monte-Carlo (RQMC)
//' 
//' This function randomly shuffles a sequence of low star discrepancy numbers in a way that preserves the low discrepancy properties according to the Scrambled Net of Matousek (1998).
//' This step is essential for RQMC, where the same low discrepancy numbers can be shuffled each iteration to induce a Central Limit Theorem and other beneficial randomness properties to the approximation
//' @param sequence an N x D matrix of N low discrepancy numbers from the D dimensional unit hypercube. Each entry should be between zero and one.
// [[Rcpp::export]]
arma::mat shuffleRQMC(arma::mat sequence){
  using namespace std;
  int N = sequence.n_rows;
  int D = sequence.n_cols;
  arma::mat output(N, D, arma::fill::zeros);
  // draw a random rule of: switch 1 and 0  /  do not switch for each binary digit.
  arma::vec rule = arma::randu<arma::vec>(16);
  for(int i = 0; i < N; ++i){
    for(int j = 0; j < D; ++j){
      // grab element of the sobol sequence
      double x = sequence(i, j);
      // convert to a binary representation
      arma::uvec binary(16, arma::fill::zeros);
      for(int k = 1; k < 17; ++k){
        if(x > pow(2, -k)){
          binary(k-1) = 1;
          x -= pow(2, -k);
        }
      }
      // apply the transform of tilde(x_k) = x_k + a_k mod 2, where a_k = 1 if rule_k > 0.5, 0 otherwise
      for(int k = 0; k < 16; ++k){
        if(rule(k) > 0.5){
          binary(k) = (binary(k) + 1) % 2;
        }
      }
      // reconstruct base 10 number from binary representation
      for(int k = 0; k < 16; ++k){
        if(binary(k) == 1){
          output(i, j) += pow(2, -(k+1));
        }
      }
    }
  }
  return output;
}