// [[Rcpp::depends(rstan)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

#include <RcppArmadillo.h>
#include <stan/math.hpp>
#include <Eigen/Dense>
#include <RcppEigen.h>
#include <Rcpp.h>

using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::MatrixXd;
using Eigen::Map;

struct RBF {
  const arma::vec y;
  const double h;
  RBF(const arma::vec& yIn, const double& hIn) :
    y(yIn), h(hIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& x)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt;

    int N = y.n_elem;

    T kernel = 0;

    for(int i = 0; i < N; ++i){
      kernel += pow(x(i) - y(i), 2);
    }

    return exp(-kernel / h);
  }
};

// [[Rcpp::export]]
Rcpp::List RBFKernel(arma::vec y, Rcpp::NumericMatrix xIn, double h){
  Map<MatrixXd> x(Rcpp::as<Map<MatrixXd> >(xIn));
  double eval;
  int dim = y.n_elem;
  Matrix<double, Dynamic, 1> grad(dim);

  RBF kernel(y, h);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(kernel, x, eval, grad);

  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = eval);
}
