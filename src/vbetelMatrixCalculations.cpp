// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

#include <RcppArmadillo.h>
#include <Rcpp.h>

//' Returns the derivative of the log-joint distribution with respect to theta for VBETEL
//' 
//' Given the intermediate inputs, this function calculates the last few stages of the derivative of the logjoint with respect to theta.
//' @param g An N x D matrix where rows correspond to the D dimensional moment condition evaluated at each of the N observations.
//' @param hessian The D x D matrix of the hessian matrix of h(theta, lambda), usually provided by nlm(...)$hessian.
//' @param lambdaHat A D dimensional vector of the optimal lambda values.
//' @param exponent An N dimensional vector of exp(lambdaHat * g) evaluated at each of the N observations
//' @param gdt A D x P x N array of the D x P matrix of the derivative of g (D dimensional) with respect to theta (P dimensional), 
//' each slice corresponds to the derivative matrix for one observation
//' @export
// [[Rcpp::export]]
Rcpp::List vbetelMatrixCalculations(arma::mat g, arma::mat hessian, arma::vec lambdaHat, arma::vec exponent, arma::cube dgdt){
  using std::pow; using std::log;
  int n = g.n_rows;
  int d = g.n_cols;
  int p = dgdt.n_cols;


  arma::mat dh2dtlam(d, p, arma::fill::zeros);
  for(int i = 0; i < n; ++i){
    dh2dtlam += exponent(i) / n * (dgdt.slice(i) + g.row(i).t() * lambdaHat.t() * dgdt.slice(i));
  }

  arma::mat dlamdt = - hessian.i() * dh2dtlam;
  arma::mat productRule(p, n, arma::fill::zeros);
  for(int i = 0; i < n; ++i){
    productRule.col(i) = dlamdt.t() * g.row(i).t() + dgdt.slice(i).t() * lambdaHat;
  }

  arma::vec dpdt = sum(productRule, 1) - n / sum(exponent) * productRule * exponent;
  
  double logp = sum(log(exponent)) - n * log(sum(exponent));
  return Rcpp::List::create(Rcpp::Named("grad") = dpdt,
                            Rcpp::Named("val") = logp);

}
