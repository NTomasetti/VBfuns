// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

#include <RcppArmadillo.h>
#include <Rcpp.h>

//' Returns the derivative of the log-joint distribution with respect to theta for VBETEL
//' 
//' Given the intermediate inputs, this function calculates the last few stages of the derivative of the logjoint with respect to theta.
//' @param g An N x P matrix where rows correspond to the P dimensional moment condition evaluated at each of the N observations.
//' @param hessian The P x P matrix of the hessian matrix of h(theta, lambda), usually provided by nlm(...)$hessian.
//' @param lambdaHat A P dimensional vector of the optimal lambda values.
//' @param exponent An N dimensional vector of exp(lambdaHat * g) evaluated at each of the N observations
//' @param gdt A P x P x N array of the P x P matrix of the derivative of g with respect to theta, each slice corresponds to one array
//' @export
// [[Rcpp::export]]
Rcpp::List vbetelMatrixCalculations(arma::mat g, arma::mat hessian, arma::vec lambdaHat, arma::vec exponent, arma::cube dgdt){
  using std::pow; using std::log;
  int n = g.n_rows;
  int d = g.n_cols;


  arma::mat dh2dtlam(d, d, arma::fill::zeros);
  for(int i = 0; i < n; ++i){
    dh2dtlam += 1.0 / n * (exponent(i) * (dgdt.slice(i) + g.row(i).t() * lambdaHat.t() * dgdt.slice(i)));
  }

  arma::mat dlamdt = - hessian.i() * dh2dtlam;
  arma::mat productRule(d, n, arma::fill::zeros);
  for(int i = 0; i < n; ++i){
    productRule.col(i) = dlamdt.t() * g.row(i).t() + dgdt.slice(i).t() * lambdaHat;
  }

  arma::vec dpdt(d, arma::fill::zeros);
  for(int j = 0; j < d; ++j){
    dpdt(j) = sum(productRule.row(j)) - n / sum(exponent) * as_scalar(productRule.row(j) * exponent);
  }
  double logp = sum(log(exponent)) - n * log(sum(exponent));
  return Rcpp::List::create(Rcpp::Named("grad") = dpdt,
                            Rcpp::Named("val") = logp);

}
