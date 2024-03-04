/* load: library(RcppEigen); library(Rcpp)  */ 
/* to compile, run: sourceCpp("funcs2.cpp") */ 

#include <boost/math/special_functions/digamma.hpp>
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
// [[Rcpp::export]]
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



// [[Rcpp::export]]
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


// [[Rcpp::export]]
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


Eigen::VectorXd rnorm(int n, double mean, double sd){
  Rcpp::NumericVector sim(n); 
  sim = Rcpp::rnorm(n, mean, sd);
  Eigen::VectorXd sample(n);
  for(int i = 0; i < n; ++i){
    sample[i] = sim[i]; 
  }
  return sample;
}



/* Make tpm from working par vector */
Eigen::MatrixXd make_tpm(int m, Eigen::VectorXd w_pars){
  /* Fill in working transition probs */
  Eigen::MatrixXd gam(m,m);
  int count = 0;
  for(int i = 0; i < m; ++i){
    gam(i,i) = 1;
    for(int j = 0; j < m; ++j){
      if(i != j){
        gam(i,j) = exp(w_pars[count]);
        count += 1;
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
double hmm_loglik(int m, Eigen::VectorXd y, Eigen::VectorXd w_pars, Eigen::RowVectorXd delta) {
  int n = y.size();
  
  /* Get parameters */ 
  Eigen::VectorXd mu = sigmoid(w_pars.segment(0*m,m)); // mean
  Eigen::VectorXd phi = w_pars.segment(1*m,m).array().square(); // disp
  Eigen::VectorXd p0 = sigmoid(w_pars.segment(2*m,m)); // zero prob
  Eigen::MatrixXd tpm = make_tpm(m, w_pars.segment( 3*m, m*(m-1) ));
  
  /* Initialization */ 
  Eigen::RowVectorXd probs = dbeta_p0_v2(m, y[0], mu, phi, p0);
  Eigen::RowVectorXd foo1 = delta.cwiseProduct(probs);
  Eigen::RowVectorXd foo2(m);
  double sumfoo = foo1.sum();
  double lscale = log(sumfoo);
  foo1 /= sumfoo;
  
  /* Scaled forward algorithm */
  for(int i = 1; i < n; ++i){
    
    /* Forward pass */
    foo2 = foo1 * tpm; //Move along the chain
    probs = dbeta_p0_v2(m, y[i], mu, phi, p0); //State densities
    foo1 = foo2.cwiseProduct(probs); // Weigh states
    
    /* Scaling and updating */
    sumfoo = foo1.sum(); // Sum elements for scaling
    lscale += log(sumfoo); // Update loglik
    foo1 /= sumfoo; // Scale forward densities
  }
  return -lscale; // Return negative log-likelihood
}

double beta_grad_mu(double y, double mu, double phi) {
  double mu_phi = mu * phi;
  double one_minus_mu_phi = (1 - mu) * phi;
  double log_y = log(y);
  double log_one_minus_y = log(1 - y);
  
  double gamma_phi = tgamma(phi);
  double gamma_mu_phi = tgamma(mu_phi);
  double gamma_one_minus_mu_phi = tgamma(one_minus_mu_phi);
  
  double pow_y_mu_phi_minus_1 = pow(y, mu_phi - 1);
  double pow_one_minus_y_one_minus_mu_phi_minus_1 = pow(1 - y, one_minus_mu_phi - 1);
  
  double digamma_mu_phi = boost::math::digamma(mu_phi);
  double digamma_one_minus_mu_phi = boost::math::digamma(one_minus_mu_phi);
  
  double term = gamma_phi * pow_y_mu_phi_minus_1 * pow_one_minus_y_one_minus_mu_phi_minus_1 / (gamma_mu_phi * gamma_one_minus_mu_phi);
  
  double res = -term * digamma_mu_phi * phi + term * digamma_one_minus_mu_phi * phi + term * phi * log_y - term * phi * log_one_minus_y;
  
  return res;
}

double beta_grad_phi(double y, double mu, double phi) {
  return ((-1 + mu)*boost::math::digamma(-(-1 + mu)*phi) + (1 - mu)*log(1 - y) + mu*log(y) - boost::math::digamma(mu*phi)*mu + boost::math::digamma(phi))*tgamma(phi)*pow(1 - y,-mu*phi + phi - 1)*pow(y,(mu*phi - 1))/(tgamma(mu*phi)*tgamma(-(-1 + mu)*phi));
}

double dbeta_cpp(double y, double mu, double phi) {
  double shape1 = mu*phi;
  double shape2 = (1-mu)*phi;
  return exp(lgamma(shape1+shape2) - lgamma(shape1) - lgamma(shape2)) * pow(y,shape1-1) * pow(1-y,shape2-1);
}

/* Function computing the gradient of zero-inflated beta for parameter i */
double beta_p0_grad(double y, double mu, double phi, double p, int i){
  double res(0);
  
  if(ISNAN(y)){ // handle missing observations
    res = 0;
  } else if(y == 0){ // y = 0
    if(i == 0){
      res = 0; 
    } else if(i == 1){
      res = 0; 
    } else if(i == 2){
      res = 1;
    }
  } else { // 0 < y < 1
    if(i == 0){
      res = (1-p)*beta_grad_mu(y,mu,phi); 
    } else if(i == 1){
      res = (1-p)*beta_grad_phi(y,mu,phi);
    } else if(i == 2){
      res = -dbeta_cpp(y,mu,phi);
    }
  }
  return res;
}


// 
// 
// /* Log-likelihood HMM */
// // [[Rcpp::export]]
// Eigen::VectorXd hmm_loglik_grad(int m, Eigen::VectorXd y, Eigen::VectorXd w_pars, Eigen::RowVectorXd delta) {
//   int n = y.size();
//   
//   /* Get parameters */ 
//   Eigen::VectorXd mu = sigmoid(w_pars.segment(0*m,m)); // mean
//   Eigen::VectorXd phi = w_pars.segment(1*m,m).array().square(); // disp
//   Eigen::VectorXd p0 = sigmoid(w_pars.segment(2*m,m)); // zero prob
//   Eigen::MatrixXd tpm = make_tpm(m, w_pars.segment( 3*m, m*(m-1) ));
//   
//   /* Initialization */ 
//   Eigen::RowVectorXd probs = dbeta_p0_v2(m, y[0], mu, phi, p0);
//   Eigen::RowVectorXd foo1 = delta.cwiseProduct(probs);
//   Eigen::RowVectorXd foo2(m);
// 
//   double sumfoo = foo1.sum();
//   double lscale = log(sumfoo);
//   foo1 /= sumfoo;
//   
//   Eigen::VectorXd grad((3*m + m*(m-1))); 
//   Eigen::MatrixXd grad_foo1(m,3*m + m*(m-1));
//   Eigen::MatrixXd grad_foo2(m,3*m + m*(m-1)); 
//   
//   bool dens_bin(false);
//   double probs_grad_trans(0); 
//   double probs_grad_dens(0);
//   
//   // int count(0);
//   // /*Score recursion for density pars*/
//   // for(int j = 0; j < m; ++j){ //loops through elements of forward densities
//   //   probs_grad_dens = beta_p0_grad(y(0), mu(j), phi(j), p0(j), 0); //gradient of density pars for parameter k modolu 3
//   //   grad_foo2(j) += probs_grad_dens*delta(j);
//   // }
//   // grad_foo1 = grad_foo2  / sumfoo;
// 
//   int count_parameter(0);
//   for(int dens_param = 0; dens_param < 3; ++dens_param){
//     for(int state = 0; state < m; ++state){
//       
//       for(int j = 0; j < m; ++j){ //loops through elements of forward densities
//         probs_grad_dens = beta_p0_grad(y(0), mu(j), phi(j), p0(j), dens_param); //gradient of density pars for parameter k modolu 3
//         double foo3 = 0;
//         int count_state(0);
//         for(int i = 0; i < m; ++i){
//           //foo3 += grad_foo1(i,count_parameter)*probs(j)*tpm(i,j); // + foo1(i)*probs(j)*probs_grad_trans ; 
//           if(state == count_state){
//             foo3 += probs_grad_dens*delta(i);
//           }
//           count_state += 1;
//         }
//         grad_foo2(j,count_parameter) = foo3 / sumfoo; 
//       }
//       count_parameter += 1;
//       
//     }
//   }
//   grad_foo1 = grad_foo2;
//   
//   
//   int par_idx(0);
//   /* Scaled forward algorithm */
//   for(int t = 1; t < n; ++t){
//     
//     probs = dbeta_p0_v2(m, y[t], mu, phi, p0); //State densities
//     
//     
//     int count_parameter(0);
//     for(int dens_param = 0; dens_param < 3; ++dens_param){
//       for(int state = 0; state < m; ++state){
//         
//         for(int j = 0; j < m; ++j){ //loops through elements of forward densities
//           probs_grad_dens = beta_p0_grad(y(t), mu(j), phi(j), p0(j), dens_param); //gradient of density pars for parameter k modolu 3
//           double foo3 = 0;
//           int count_state(0);
//           for(int i = 0; i < m; ++i){
//             foo3 += grad_foo1(i,count_parameter)*probs(j)*tpm(i,j); // + foo1(i)*probs(j)*probs_grad_trans ; 
//             if(state == count_state){
//               foo3 += foo1(i)*probs_grad_dens*tpm(i,j);
//             }
//             count_state += 1;
//           }
//           grad_foo2(j,count_parameter) = foo3 / sumfoo; 
//         }
//         count_parameter += 1;
//         
//       }
//     }
//     grad_foo1 = grad_foo2; // / sumfoo; 
//     
//     
//     //     int count(0); 
//     //     /*Score recursion for density pars*/
//     //     for(int j = 0; j < m; ++j){ //loops through elements of forward densities
//     //       probs_grad_dens = beta_p0_grad(y(t), mu(j), phi(j), p0(j), 0); //gradient of density pars for parameter k modolu 3
//     //       double foo3 = 0;
//     //       for(int i = 0; i < m; ++i){
//     //         foo3 += (grad_foo1(i)*probs(j)*tpm(i,j) + ((i+1) % 2 )*foo1(i)*probs_grad_dens*tpm(i,j)); // + foo1(i)*probs(j)*probs_grad_trans ; 
//     //       }
//     //       grad_foo2(j) = foo3 / sumfoo; 
//     //     }
//     //     grad_foo1 = grad_foo2; // / sumfoo; 
//     
//     /* Forward pass */
//     foo2 = foo1 * tpm; //Move along the chain
//     foo1 = foo2.cwiseProduct(probs); // Weigh states
// 
//     /* Scaling and updating */
//     sumfoo = foo1.sum(); // Sum elements for scaling
//     lscale += log(sumfoo); // Update loglik
//     foo1 /= sumfoo; // Scale forward densities
//     
//   }
//   for(int i = 0; i<(3*m + m*(m-1)); ++i){
//     grad(i) = grad_foo1.col(i).sum() / sumfoo;
//   }
//   return -grad; // Return negative log-likelihood
// }



// [[Rcpp::export]]
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


// [[Rcpp::export]]
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

// [[Rcpp::export]]
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



/* HMM predictions */
// [[Rcpp::export]]
Rcpp::List hmm_forecast(int m, Eigen::VectorXd y, Eigen::VectorXd w_pars, Eigen::RowVectorXd delta, int ahead, int n_sim = 10000) {
  int n = y.size();
  
  /* Get parameters */ 
  Eigen::VectorXd mu = sigmoid(w_pars.segment(0*m,m)); // mean
  Eigen::VectorXd phi = w_pars.segment(1*m,m).array().square(); // disp
  Eigen::VectorXd p0 = sigmoid(w_pars.segment(2*m,m)); // zero prob
  Eigen::MatrixXd tpm = make_tpm(m, w_pars.segment( 3*m, m*(m-1) ));
  
  /* Initialization */ 
  Eigen::RowVectorXd probs = dbeta_p0_v2(m, y[0], mu, phi, p0);
  Eigen::RowVectorXd foo = delta.cwiseProduct(probs);
  Eigen::RowVectorXd foo_alias = delta;
  Eigen::RowVectorXd u(m);
  double sumfoo = foo.sum();
  foo /= sumfoo;
  Eigen::MatrixXd tpm_ahead = tpm.pow(ahead);
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
  
  return L; 
}


/* AR(1) predictions */
// [[Rcpp::export]]
Rcpp::List AR1_forecast(Eigen::VectorXd y, int ahead, double c_AR, double sig2_AR, int n_sim = 10000 ) {

  /* Initialization */ 
  int n = y.size();
  double persist(0);
  int h = ahead;
  Eigen::VectorXd AR_sim(n_sim);
  double crps_AR(0);
  int n_no_nan(0);
  Eigen::VectorXd E_AR(n); 
  Eigen::VectorXd V_AR(n);
  Eigen::VectorXd Q1(n); 
  Eigen::VectorXd Q2(n);
  Eigen::VectorXd Q3(n);
  Eigen::VectorXd pseudo(n);
  

  /* Iterate forward */
  for(int i = 1; i < n; ++i){

    /* Prediction */
    if( (i % ahead) == 0){
      h = ahead;
      if(!ISNAN(y[i-ahead])){
        persist = y[i-ahead];
      }
    } else {
      h += 1; 
    }
    
    /* Predicted density parameters */
    E_AR(i) = pow(c_AR, h)*persist; // Expectation of AR(1)
    V_AR(i) = sig2_AR * ( (1 - pow(c_AR, 2*h) )  / ( 1 - pow(c_AR, 2) )); // Variance of AR(1)
    
    /* Sample from predictive distribution */
    AR_sim = rnorm(n_sim, E_AR(i), sqrt(V_AR(i)));
    
    /* Evaluation */
    if(!ISNAN(y[i])){
      n_no_nan += 1;
      crps_AR += crps_cpp(AR_sim, y[i], false);
      pseudo(i) = R::pnorm( y[i], E_AR(i), sqrt(V_AR(i)), true, false );
    } else {pseudo(i) = R_NaN; }
    
    /* Compute quartiles */
    std::sort(AR_sim.data(), AR_sim.data() + n);
    Q1[i] = AR_sim(static_cast<int>(n * 0.25));
    Q2[i] = AR_sim(static_cast<int>(n * 0.50));
    Q3[i] = AR_sim(static_cast<int>(n * 0.75));
  }
  
  Rcpp::List L = Rcpp::List::create(Rcpp::Named("CRPS_AR") = crps_AR/n_no_nan, Rcpp::_["Mean"] = E_AR, Rcpp::_["Variance"] = V_AR, Rcpp::_["Q1"] = Q1, Rcpp::_["Q2"] = Q2, Rcpp::_["Q3"] = Q3, Rcpp::_["pseudo"] = pseudo);
  
  return L; 
}


























