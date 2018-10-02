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

/* stan::math automatic differentiation requires two components: A structure containing the object to be differentiated,
* and a function that interfaces with R and calls stan::math::gradient on the initialised structure. 
* The structure has a constructer that inputs the non-differentiated parameter of any type,
*  and an operator () that inputs the parameter that is differentiated which must be an Eigen::Matrix.
* To help identify parameters, Eigen objects are used exclusively for () inputs and arma objects for other vectors / matrices etc. that are constructor inputs.
*/
struct RBF {
  const arma::vec y;
  const double h;
  RBF(const arma::vec& yIn, const double& hIn) :
    y(yIn), h(hIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& x)
    const{
    using std::exp; using std::pow;

    int N = y.n_elem;

    T kernel = 0;
    for(int i = 0; i < N; ++i){
      kernel += pow(x(i) - y(i), 2);
    }

    return exp(-kernel / h);
  }
};

//' Evaluates and differentiates the RBF Kernel.
//' 
//' This function returns a list of the value of the RBF Kernel, and gradient with respect to the second argument, of the RBF Kernel.
//' @param y Vector of inputs that are not differentiated.
//' @param xIn N * 1 matrix of inputs that are differentiated.
//' @param h Smoothing parameter.
// [[Rcpp::export]]
Rcpp::List RBFKernel(arma::vec y, Rcpp::NumericMatrix xIn, double h){
  // We cannot directly input Eigen objects into Rcpp functions, but must instead map an Rcpp object to an Eigen object.
  Map<MatrixXd> x(Rcpp::as<Map<MatrixXd> >(xIn));
  // The value of the kernel is stored in eval.
  double eval;
  // Set up the gradient vector, of dimension equal to xIn.
  int dim = y.n_elem;
  Matrix<double, Dynamic, 1> grad(dim);
  // Initialise the RBF kernel structure with non-differntiated parameters.
  RBF kernel(y, h);
  // Reset any internal object in stan to zero.
  stan::math::set_zero_all_adjoints();
  // Take the gradient of kernel with respect to x, store the value in eval and the gradient vector in grad.
  stan::math::gradient(kernel, x, eval, grad);
  // Return as an R List.
  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = eval);
}
