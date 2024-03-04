/* load: library(RcppEigen); library(Rcpp)  */ 
/* to compile, run: sourceCpp("funcs2.cpp") */ 

#include <unsupported/Eigen/MatrixFunctions>
#include <RcppEigen.h>
#include <Rcpp.h>
#include <algorithm>


// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppEigen)]]



/* Parameter transform from R -> (0,1), in 2 vers. (overloading allowed)*/ 
double sigmoid(double x){
  return exp(x) / (exp(x) + 1 );
}
Eigen::VectorXd sigmoid(Eigen::VectorXd x){
  return x.array().exp() / (x.array().exp() + 1);
}


/* Parameter transform from R -> (0,1)*/
double logit(double x){
  return log(x / (1 - x));
}


/* Zero-inflated beta density */
Eigen::VectorXd dbeta_p0(Eigen::VectorXd x, double mu, double phi, double p0){
  double shape1 = mu*phi;
  double shape2 = (1-mu)*phi;
  double foo(0);
  int n = x.size();
  Eigen::VectorXd res(n);
  
  for(int i = 0; i < n; ++i) { // loop through all data points
    if(x[i] == 0) {
      res[i] = p0; // probaility mass in x = 0
    }
    else {
      foo = exp(lgamma(shape1+shape2) - lgamma(shape1) - lgamma(shape2)) * pow(x[i],shape1-1) * pow(1-x[i],shape2-1);
      res[i] = (1-p0)*foo; // assign beta density corrected with the zero prob mass
    }
  }
  return res;
}



Eigen::RowVectorXd dbeta_p0_v2(int m, double x, Eigen::VectorXd mu, Eigen::VectorXd phi, Eigen::VectorXd p0){
  double shape1(0);
  double shape2(0);
  double foo(0);
  Eigen::VectorXd res(m);
  
  for(int i = 0; i < m; ++i) { 
    if(ISNAN(x)){ // handle missing observations
      res[i] = 1;
    } else if(x == 0) {
      res[i] = p0[i]; // probaility mass in x = 0
    } else {
      shape1 = mu[i]*phi[i];
      shape2 = (1-mu[i])*phi[i];
      foo = exp(lgamma(shape1+shape2) - lgamma(shape1) - lgamma(shape2)) * pow(x,shape1-1) * pow(1-x,shape2-1);
      res[i] = (1-p0[i])*foo; // assign beta density corrected with the zero prob mass
    }
  }
  return res;
}


double dbeta_p0_v2(int m, double x, Eigen::VectorXd mu, Eigen::VectorXd phi, Eigen::VectorXd p0, Eigen::RowVectorXd u){
  double shape1(0);
  double shape2(0);
  double foo(0);
  Eigen::VectorXd res(m);
  
  for(int i = 0; i < m; ++i) { 
    if(ISNAN(x)){ // handle missing observations
      res[i] = 1;
    } else if(x == 0) {
      res[i] = p0[i]; // probaility mass in x = 0
    } else {
      shape1 = mu[i]*phi[i];
      shape2 = (1-mu[i])*phi[i];
      foo = exp(lgamma(shape1+shape2) - lgamma(shape1) - lgamma(shape2)) * pow(x,shape1-1) * pow(1-x,shape2-1);
      res[i] = (1-p0[i])*foo; // assign beta density corrected with the zero prob mass
    }
  }
  return res.dot(u);
}


Rcpp::IntegerVector seq_cpp(int lo, int hi) {
  int n = hi - lo + 1;
  
  // Create a new integer vector, sequence, of size n
  Rcpp::IntegerVector sequence(n);
  for(int i = 0; i < n; i++) {
    // Set the ith element of sequence to lo plus i
    sequence[i] = lo + i;
  }
  return sequence;
}


bool nextBool(double probability){
  return rand() <  probability * ((double)RAND_MAX + 1.0);
}


Eigen::VectorXd rbeta_p0(int n, int m, Eigen::VectorXd mu, Eigen::VectorXd phi, Eigen::VectorXd p0, Eigen::RowVectorXd u){
  // idea:
  // sample state j from 1 to m, with probs given by u_j.
  // Then sample 0 with prob p0_j and sample 1 with prob 1-p0_j i.e.: bool foo = (rand() % 100) < (p0*100)
  // if sample 0, then x = 0; if sample 1, then x = beta(mu_j,phi_j)
  
  /* Initialization */
  Eigen::VectorXd shape1(m);
  Eigen::VectorXd shape2(m);
  double foo(0);
  Eigen::VectorXd sample(n);
  bool zero_sample(false);
  int j(0); //state counter
  Rcpp::NumericVector u_probs(m); 
  
  /* Compute density pars */
  for(int i = 0; i < m; ++i) { 
    shape1[i] = mu[i]*phi[i];
    shape2[i] = (1-mu[i])*phi[i];
    u_probs[i] = u[i];
  }
  
  /* Sample states */
  Rcpp::IntegerVector state_space = seq_cpp(0,m-1); // Create state vector
  Rcpp::IntegerVector state_sample = Rcpp::sample(state_space, n, true, u_probs); // sample state
  
  /* Generate n draws */
  for(int i = 0; i < n; ++i){
    j = state_sample[i];
    zero_sample = nextBool(p0[j]);
    if(zero_sample){
      sample[i] = 0;
    } else {
      sample[i] = R::rbeta(shape1[j], shape2[j]); 
    }
    
  }
  return sample;
}


Eigen::MatrixXd make_tpm(int m, Eigen::RowVectorXd w_pars_base, Eigen::RowVectorXd w_pars_cov, Eigen::RowVectorXd covs){
  Eigen::MatrixXd gam(m,m);
  int n_covs = covs.size(); 
  
  int count(0);
  int count2(0);
  for(int i = 0; i < m; ++i){
    gam(i,i) = 1;
    for(int j = 0; j < m; ++j){
      if(i != j){
        gam(i,j) = exp(w_pars_base[count] + covs.dot(w_pars_cov.segment(count2,n_covs) ) ) ;
        count += 1;
        count2 += n_covs;
      }
    }
  }
  /* Scale rows by row sums */ 
  double rowScale(0);
  for(int i = 0; i < m; ++i){
    rowScale = 0;
    for(int j = 0; j < m; ++j){
      rowScale += gam(i,j);
      if(j == (m-1)){
        for(int k = 0; k < m; ++k){ gam(i,k) /= rowScale; }
      }
    }
  }
  return gam;
}

/* Log-likelihood HMM */
// [[Rcpp::export]]
double hmm_loglik_Transcovs(int m, Eigen::VectorXd y, Eigen::VectorXd w_pars, Eigen::RowVectorXd delta, Eigen::MatrixXd covs) {
  int n = y.size();
  int n_covs = covs.cols();
  double res(0);
  
  /* Get parameters */ 
  Eigen::VectorXd mu = sigmoid(w_pars.segment(0*m,m)); // mean
  Eigen::VectorXd phi = w_pars.segment(1*m,m).array().square(); // disp
  Eigen::VectorXd p0 = sigmoid(w_pars.segment(2*m,m)); // zero prob
  Eigen::RowVectorXd Transbase = w_pars.segment( 3*m, m*(m-1) ).transpose();
  Eigen::RowVectorXd Transcovs = w_pars.segment( 3*m + m*(m-1), n_covs*m*(m-1) ).transpose();
  
  
  /* Initialization */ 
  Eigen::RowVectorXd probs = dbeta_p0_v2(m, y[0], mu, phi, p0);
  Eigen::RowVectorXd foo1 = delta.cwiseProduct(probs);
  Eigen::RowVectorXd foo2(m);
  Eigen::MatrixXd tpm(m,m);
  double sumfoo = foo1.sum();
  double lscale = log(sumfoo);
  foo1 /= sumfoo;
  
  /* Scaled forward algorithm */
  for(int i = 1; i < n; ++i){
    
    tpm = make_tpm(m, Transbase, Transcovs, covs.row(i-1));
    
    //int m, Eigen::VectorXd w_pars_base, Eigen::VectorXd w_pars_cov, Eigen::VectorXd covs
    
    /* Forward pass */
    foo2 = foo1 * tpm; //Move along the chain
    probs = dbeta_p0_v2(m, y[i], mu, phi, p0); //State densities
    foo1 = foo2.cwiseProduct(probs); // Weigh states
    
    /* Scaling and updating */
    sumfoo = foo1.sum(); // Sum elements for scaling
    lscale += log(sumfoo); // Update loglik
    foo1 /= sumfoo; // Scale forward densities
  }
  
  /*Handle NaNs*/
  if(ISNAN(lscale)){
    res = 100000;
  } else{
    res = -lscale;
  }
  return res; // Return negative log-likelihood
}


Eigen::VectorXd ecdf_cpp(Eigen::VectorXd sim, bool sorted = false) {
  int n = sim.size();
  
  /* Sorting */
  Eigen::VectorXd sorted_sim = sim;
  if(!sorted){ std::sort(sorted_sim.data(), sorted_sim.data() + n);}
  
  /* Initialization */
  double t = sorted_sim[0]; //start value
  double res = (sorted_sim[n-1] - sorted_sim[0]) / (n-1); //resolution
  int cdf_acu(0); 
  int start(0);
  Eigen::VectorXd cdf(n);
  
  /* Cumulative indicator sum on sorted array */
  for (int i = 0; i < n; ++i) {
    for (int j = start; j < n; ++j){
      if(sorted_sim[j] <= t){
        cdf_acu += 1; 
      } else { break; }
    }
    start = cdf_acu; 
    cdf[i] = cdf_acu;
    t += res;
  }
  return cdf / static_cast<double>(n); 
}


double pseudo_residual(double x, Eigen::VectorXd sim, bool sorted = false) {
  int n = sim.size();
  
  /* Sorting */
  Eigen::VectorXd sorted_sim = sim;
  if(!sorted){ std::sort(sorted_sim.data(), sorted_sim.data() + n);}
  
  /* Initialization */
  double t = sorted_sim[0]; //start value
  double res = (sorted_sim[n-1] - sorted_sim[0]) / (n-1); //resolution
  int cdf_acu(0); 
  int start(0);
  Eigen::VectorXd cdf(n);
  double foo(0);
  double min_diff(1);
  int min_idx(0);
  
  /* Cumulative indicator sum on sorted array */
  for (int i = 0; i < n; ++i) {
    /* Count values up to evaluation point */
    for (int j = start; j < n; ++j){
      if(sorted_sim[j] <= t){
        cdf_acu += 1; 
      } else { break; }
    }
    start = cdf_acu; 
    cdf[i] = cdf_acu;
    
    /* Find closest evalutation point to x */
    foo = abs(x-t);
    if(foo < min_diff){
      min_idx = i;
      min_diff = foo;
    }
    
    /* Update evalutation point */
    t += res;
  }
  return cdf(min_idx) / n;
}


int Heaviside(double x) {
  int res = 1;
  if(x < 0){
    res = 0;
  }
  return res; 
}

double crps_cpp(Eigen::VectorXd sim, double y, bool sorted = false) {
  int n = sim.size();
  
  /* Sorting */
  Eigen::VectorXd sorted_sim = sim;
  if(!sorted){ std::sort(sorted_sim.data(), sorted_sim.data() + n);}
  
  /* Empirical cdf */ 
  Eigen::VectorXd ecdf = ecdf_cpp(sorted_sim, true); 
  
  /* Initialization */
  double t = sorted_sim[0]; //start value
  double res = (sorted_sim[n-1] - sorted_sim[0]) / (n-1); //resolution
  double area(0);
  
  /* Approximate crps integral by rectangular rule */
  for (int i = 0; i < n; ++i) {
    area += pow(ecdf[i] - Heaviside(t - y), 2);
    t += res;
  }
  return area*res; 
}


// [[Rcpp::export]]
Rcpp::List hmm_forecast_Transcovs(int m, Eigen::VectorXd y, Eigen::VectorXd w_pars, Eigen::RowVectorXd delta, Eigen::MatrixXd covs, int ahead, int n_sim = 10000 ) {
  int n = y.size();
  int n_covs = covs.cols();
  
  /* Get parameters */ 
  Eigen::VectorXd mu = sigmoid(w_pars.segment(0*m,m)); // mean
  Eigen::VectorXd phi = w_pars.segment(1*m,m).array().square(); // disp
  Eigen::VectorXd p0 = sigmoid(w_pars.segment(2*m,m)); // zero prob
  Eigen::RowVectorXd Transbase = w_pars.segment( 3*m, m*(m-1) ).transpose();
  Eigen::RowVectorXd Transcovs = w_pars.segment( 3*m + m*(m-1), n_covs*m*(m-1) ).transpose();
  
  /* Initialization */
  Eigen::MatrixXd tpm(m,m);
  Eigen::MatrixXd tpm_ahead(m,m);
  Eigen::RowVectorXd probs = dbeta_p0_v2(m, y[0], mu, phi, p0);
  Eigen::RowVectorXd foo = delta.cwiseProduct(probs);
  Eigen::RowVectorXd foo_alias = delta;
  Eigen::RowVectorXd u(m);
  double sumfoo = foo.sum();
  foo /= sumfoo;
  int h = ahead;
  Eigen::VectorXd HMM_sim(n_sim);
  double crps_HMM(0); 
  int n_no_nan(0);
  Eigen::VectorXd Q1(n); 
  Eigen::VectorXd Q2(n);
  Eigen::VectorXd Q3(n);
  Eigen::MatrixXd u_stored(n,m);
  Eigen::VectorXd pseudo(n);
  
  /* Scaled forward algorithm */
  for(int i = 1; i < n; ++i){
    
    tpm = make_tpm(m, Transbase, Transcovs, covs.row(i-1));
    tpm_ahead = tpm.pow(ahead);
    
    /* Prediction */
    if( (i % ahead) == 0){ 
      u = foo * tpm_ahead; 
      h = ahead;
    } else {
      u = u_stored.row(i-1) * tpm;
      h += 1; 
    }
    
    /* Forward pass */
    foo_alias = foo * tpm; //Move along the chain
    probs = dbeta_p0_v2(m, y[i], mu, phi, p0); //State densities
    foo = foo_alias.cwiseProduct(probs); // Weigh states
    
    /* Scaling and updating */
    sumfoo = foo.sum(); // Sum elements for scaling
    foo /= sumfoo; // Scale forward densities
    
    /* Density weights */
    u_stored.row(i) = u;
    
    /* Sample from predictive distribution */
    HMM_sim = rbeta_p0(n_sim, m, mu, phi, p0, u);
    
    /* Evaluation */
    if(!ISNAN(y[i])){
      n_no_nan += 1;
      crps_HMM += crps_cpp(HMM_sim, y[i], false);
      pseudo(i) = pseudo_residual(y[i], HMM_sim);
    } else {pseudo(i) = R_NaN; }
    
    /* Compute quartiles */
    std::sort(HMM_sim.data(), HMM_sim.data() + n);
    Q1[i] = HMM_sim(static_cast<int>(n * 0.25));
    Q2[i] = HMM_sim(static_cast<int>(n * 0.50));
    Q3[i] = HMM_sim(static_cast<int>(n * 0.75));
    
  }
  
  Rcpp::List L = Rcpp::List::create(Rcpp::Named("CRPS_HMM") = crps_HMM/n_no_nan , Rcpp::_["mu"] = mu, Rcpp::_["phi"] = phi, Rcpp::_["p0"] = p0, Rcpp::_["u"] = u_stored, Rcpp::_["Q1"] = Q1, Rcpp::_["Q2"] = Q2, Rcpp::_["Q3"] = Q3, Rcpp::_["pseudo"] = pseudo);
  
  return L; // Return negative log-likelihood
}



