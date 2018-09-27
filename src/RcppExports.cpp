// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// RBFKernel
Rcpp::List RBFKernel(arma::vec y, Rcpp::NumericMatrix xIn, double h);
RcppExport SEXP _VBfuns_RBFKernel(SEXP ySEXP, SEXP xInSEXP, SEXP hSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type xIn(xInSEXP);
    Rcpp::traits::input_parameter< double >::type h(hSEXP);
    rcpp_result_gen = Rcpp::wrap(RBFKernel(y, xIn, h));
    return rcpp_result_gen;
END_RCPP
}
// vbetelMatrixCalculations
Rcpp::List vbetelMatrixCalculations(arma::mat g, arma::mat hessian, arma::vec lambdaHat, arma::vec exponent, arma::cube dgdt);
RcppExport SEXP _VBfuns_vbetelMatrixCalculations(SEXP gSEXP, SEXP hessianSEXP, SEXP lambdaHatSEXP, SEXP exponentSEXP, SEXP dgdtSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type g(gSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type hessian(hessianSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambdaHat(lambdaHatSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type exponent(exponentSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type dgdt(dgdtSEXP);
    rcpp_result_gen = Rcpp::wrap(vbetelMatrixCalculations(g, hessian, lambdaHat, exponent, dgdt));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_VBfuns_RBFKernel", (DL_FUNC) &_VBfuns_RBFKernel, 3},
    {"_VBfuns_vbetelMatrixCalculations", (DL_FUNC) &_VBfuns_vbetelMatrixCalculations, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_VBfuns(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
