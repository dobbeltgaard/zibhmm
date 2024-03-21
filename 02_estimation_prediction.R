rm(list = ls())


###########################
### Read code libraries ###
###########################

d <- read.csv("ice_data.csv")

library(RcppEigen); library(Rcpp); library("zoo"); library("forecast");
sourceCpp("func_hmm.cpp")
sourceCpp("func_hmm_transcov.cpp")
sourceCpp("func_hmm_denscov.cpp")

cov_names = c("cbh", "t");
Z = matrix(NA, nrow = NROW(d), ncol = length(cov_names))
for(i in 1:length(cov_names)){
  Z[,i] = matrix(as.matrix(d[, cov_names[i]]), ncol = length(cov_names[i])) * (1/apply(  abs(matrix(as.matrix(d[, cov_names]), ncol = length(cov_names))), 2, max))[i]
}



################################
### Estimation & Forecasting ###
################################

###################
### AR(1) model ###
###################
ts_dat = ts(d$icing, frequency = 1)
ts_dat = na.approx(ts_dat) #linear approx of missing values
ar.1 = arima(ts_dat[d$wint1], order = c(1,0,0), include.mean = F)

predAR_w2 = AR1_forecast(y = d$icing.trans[d$wint2], ahead = 12, c_AR = ar.1$coef, sig2_AR = ar.1$sigma2 )
predAR_w3 = AR1_forecast(y = d$icing.trans[d$wint3], ahead = 12, c_AR = ar.1$coef, sig2_AR = ar.1$sigma2 )
predAR_w2$CRPS_AR #CRPS = 0.07407545
predAR_w3$CRPS_AR #CRPS = 0.07092122
mean(abs(predAR_w2$Q2 - d$icing.trans[d$wint2]), na.rm = T)
mean(abs(predAR_w3$Q2 - d$icing.trans[d$wint3]), na.rm = T)


mean(abs(d$icing.trans[d$wint3]), na.rm = T)

save(predAR_w2, predAR_w3, file = "ar_models_preds.RData")


###################
### ARIMA AUTO ####
###################
arimafit <- auto.arima(ts_dat[d$wint1])
forecast(arimafit,h = 12)


###################
### Linear Reg ####
###################
data = data.frame(d$icing, Z)
linfit <- lm(d.icing ~ X1 + X2, data = data[d$wint1, ])
linfit$coefficients
linpredw2 <- predict(linfit, data[d$wint2, ])
linpredw3 <- predict(linfit, data[d$wint3, ])

mean(abs(linpredw2 - d$icing.trans[d$wint2]), na.rm = T)
mean(abs(linpredw3 - d$icing.trans[d$wint3]), na.rm = T)

###################
### beta ZI reg ###
###################
mu_init = c(0.2); 
phi_init = c(4);
p0_init = c(0.3); 
mucov_init = rep(1, length(cov_names))
phicov_init = rep(1, length(cov_names))
p0cov_init = rep(1, length(cov_names))
phi_covs = T; p0_covs = T;
par0 = c(mu_init, phi_init, p0_init, mucov_init, phicov_init, p0cov_init);
#par0 = c(mu_init, phi_init, p0_init, mucov_init, gam_init);
#hmm_loglik_Denscovs(m = m, y =d$icing.trans[d$wint1], w_pars = par0,delta = rep(1,m)/m, covs = matrix(as.matrix(d[d$wint1, cov_names]), ncol = length(cov_names)), phi_covs = phi_covs, p0_covs = p0_covs )
betareg <- optim(par = par0, fn = beta_p0_loglik, y = d$icing.trans[d$wint1 & !is.na(d$icing.trans)], covs = Z[d$wint1 & !is.na(d$icing.trans), ], 
                 phi_covs = phi_covs, p0_covs = p0_covs, gr = NULL, method = "BFGS", control = list(maxit = 1000))
length(betareg$par) # = 9 = number of estimated parameters
2 * (betareg$value + length(betareg$par)) #AIC = 
predbeta_w2 = beta_p0_forecast(y = d$icing.trans[d$wint2], w_pars = betareg$par, covs = Z[d$wint2, ], phi_covs = phi_covs, p0_covs = p0_covs)
predbeta_w3 = beta_p0_forecast(y = d$icing.trans[d$wint3], w_pars = betareg$par, covs = Z[d$wint3, ], phi_covs = phi_covs, p0_covs = p0_covs)
predbeta_w2$CRPS
predbeta_w3$CRPS
mean(abs(predbeta_w2$Q2 - d$icing.trans[d$wint2]), na.rm = T)
mean(abs(predbeta_w3$Q2 - d$icing.trans[d$wint3]), na.rm = T)


###################
### HMM no covs ###
###################
# 2-state model
m = 2;
mu_init = rep(0.2,m); 
phi_init = rep(3, m);
p0_init = rep(0.3, m); 
factk = 0.75; gam0 = diag(m) * factk; gam0[upper.tri(gam0) | lower.tri(gam0)] = (1-factk)/(m-1);
gam_init = gam0[!c(diag(m))];
par0 = c(mu_init, phi_init, p0_init, gam_init);
r2 <- optim(par = par0, fn = hmm_loglik, delta = rep(1,m)/m, m = m,y = d$icing.trans[d$wint1], gr = NULL, method = "BFGS")
2 * (r2$value + length(r2$par)) #AIC = 409.9768
predr2_w2 = hmm_forecast(m = m, y = d$icing.trans[d$wint2], w_pars = r2$par, delta = rep(1,m)/m,ahead = 12 )
predr2_w3 = hmm_forecast(m = m, y = d$icing.trans[d$wint3], w_pars = r2$par, delta = rep(1,m)/m,ahead = 12 )
predr2_w2$CRPS_HMM #CRPS = 0.06592973
predr2_w3$CRPS_HMM #CRPS = 0.05438384


hmm_loglik(m = m, y = d$icing.trans[d$wint1], w_pars = par0, delta = rep(1,m)/m)
hmm_loglik_grad2(m = m, y = d$icing.trans[d$wint1], w_pars = par0, delta = rep(1,m)/m)
pracma::grad(f = hmm_loglik, x0 = par0, m = m, y = d$icing.trans[d$wint1], delta = rep(1,m)/m )
pracma::grad(f = hmm_loglik, x0 = par0, m = m, y = d$icing.trans[d$wint1], delta = rep(1,m)/m )


#3-state model 
m = 3;
mu_init = rep(0.1,m); 
phi_init = rep(2, m);
p0_init = rep(0.3, m); 
factk = 0.75; gam0 = diag(m) * factk; gam0[upper.tri(gam0) | lower.tri(gam0)] = (1-factk)/(m-1);
gam_init = gam0[!c(diag(m))];
par0 = c(mu_init, phi_init, p0_init, gam_init);
r3 <- optim(par = par0, fn = hmm_loglik, delta = rep(1,m)/m, m = m,y = d$icing.trans[d$wint1], gr = NULL, method = "BFGS")
2 * (r3$value + length(r3$par)) #AIC = -584.3532
predr3_w2 = hmm_forecast(m = m, y = d$icing.trans[d$wint2], w_pars = r3$par, delta = rep(1,m)/m,ahead = 12 )
predr3_w3 = hmm_forecast(m = m, y = d$icing.trans[d$wint3], w_pars = r3$par, delta = rep(1,m)/m,ahead = 12 )
predr3_w2$CRPS_HMM #CRPS = 0.0554503
predr3_w3$CRPS_HMM #CRPS = 0.04516442

#4-state model 
m = 4;
mu_init = rep(0.1,m); 
phi_init = rep(2, m);
p0_init = rep(0.3, m); 
factk = 0.75; gam0 = diag(m) * factk; gam0[upper.tri(gam0) | lower.tri(gam0)] = (1-factk)/(m-1);
gam_init = gam0[!c(diag(m))];
par0 = c(mu_init, phi_init, p0_init, gam_init);
r4 <- optim(par = par0, fn = hmm_loglik, delta = rep(1,m)/m, m = m,y = d$icing.trans[d$wint1], 
            gr = NULL, method = "BFGS", control = list(maxit = 1000))
2 * (r4$value + length(r4$par)) #AIC = -740.0393
predr4_w2 = hmm_forecast(m = m, y = d$icing.trans[d$wint2], w_pars = r4$par, delta = rep(1,m)/m,ahead = 12 )
predr4_w3 = hmm_forecast(m = m, y = d$icing.trans[d$wint3], w_pars = r4$par, delta = rep(1,m)/m,ahead = 12 )
predr4_w2$CRPS_HMM #CRPS = 0.04905068
predr4_w3$CRPS_HMM #CRPS = 0.04299427



#5-state model !!!
m = 5;
mu_init = rep(0.1,m); 
phi_init = rep(2, m);
p0_init = rep(0.3, m); 
factk = 0.75; gam0 = diag(m) * factk; gam0[upper.tri(gam0) | lower.tri(gam0)] = (1-factk)/(m-1);
gam_init = gam0[!c(diag(m))];
par0 = c(mu_init, phi_init, p0_init, gam_init);
r5 <- optim(par = par0, fn = hmm_loglik, delta = rep(1,m)/m, m = m,y = d$icing.trans[d$wint1], 
            gr = NULL, method = "BFGS", control = list(maxit = 1000), hessian = T)
2 * (r5$value + length(r5$par)) #AIC = -1534.46
length(r5$par) # = 35 = number of estimated parameters
predr5_w2 = hmm_forecast(m = m, y = d$icing.trans[d$wint2], w_pars = r5$par, delta = rep(1,m)/m,ahead = 12 )
predr5_w3 = hmm_forecast(m = m, y = d$icing.trans[d$wint3], w_pars = r5$par, delta = rep(1,m)/m,ahead = 12 )
predr5_w2$CRPS_HMM #CRPS = 0.04814774
predr5_w3$CRPS_HMM #CRPS = 0.04084666

mean(abs(predr5_w2$Q2 - d$icing.trans[d$wint2]), na.rm = T) #MAE = 0.06152247
mean(abs(predr5_w3$Q2 - d$icing.trans[d$wint3]), na.rm = T) #MAE = 0.05032206


save(r5, predr5_w2, predr5_w3, file = "hmm5_models_preds.RData")
make_tpm(5, r5$par[-(1:15)])


#6-state model 
m = 6;
mu_init = rep(0.1,m); 
phi_init = rep(2, m);
p0_init = rep(0.3, m); 
factk = 0.75; gam0 = diag(m) * factk; gam0[upper.tri(gam0) | lower.tri(gam0)] = (1-factk)/(m-1);
gam_init = gam0[!c(diag(m))];
par0 = c(mu_init, phi_init, p0_init, gam_init);
r6 <- optim(par = par0, fn = hmm_loglik, delta = rep(1,m)/m, m = m,y = d$icing.trans[d$wint1], 
            gr = NULL, method = "BFGS", control = list(maxit = 1000))
2 * (r6$value + length(r6$par)) #AIC = 366.1866
predr6_w2 = hmm_forecast(m = m, y = d$icing.trans[d$wint2], w_pars = r6$par, delta = rep(1,m)/m,ahead = 12 )
predr6_w3 = hmm_forecast(m = m, y = d$icing.trans[d$wint3], w_pars = r6$par, delta = rep(1,m)/m,ahead = 12 )
predr6_w2$CRPS_HMM #CRPS = 0.06503083
predr6_w3$CRPS_HMM #CRPS = 0.05306617


######################
### HMM trans covs ###
######################
cov_names = c("cbh", "t");

#2-state model
m = 2;
mu_init = rep(0.1,m); 
phi_init = rep(2, m);
p0_init = rep(0.3, m); 
factk = 0.75; gam0 = diag(m) * factk; gam0[upper.tri(gam0) | lower.tri(gam0)] = (1-factk)/(m-1);
gam_init = gam0[!c(diag(m))];
gamcov_init = rep(2.5, length(cov_names)*length(gam_init));
par0 = c(mu_init, phi_init, p0_init, gam_init, gamcov_init);
r2transcov <- optim(par = par0, fn = hmm_loglik_Transcovs, delta = rep(1,m)/m, m = m,y = d$icing.trans[d$wint1], 
               covs = matrix(as.matrix(d[d$wint1, cov_names]), ncol = length(cov_names)), gr = NULL, method = "BFGS", control = list(maxit = 1000))
2 * (r2transcov$value + length(r2transcov$par)) #AIC = 405.4716
predr2transcov_w2 = hmm_forecast_Transcovs(m = m, y = d$icing.trans[d$wint2], w_pars = r2transcov$par, delta = rep(1,m)/m, 
                                           covs = matrix(as.matrix(d[d$wint2, cov_names]), ncol = length(cov_names)), ahead = 12 )
predr2transcov_w3 = hmm_forecast_Transcovs(m = m, y = d$icing.trans[d$wint3], w_pars = r2transcov$par, delta = rep(1,m)/m, 
                                           covs = matrix(as.matrix(d[d$wint3, cov_names]), ncol = length(cov_names)), ahead = 12 )
predr2transcov_w2$CRPS_HMM #CRPS = 0.06352258
predr2transcov_w3$CRPS_HMM #CRPS = 0.05362285

#3-state model !!
m = 3;
mu_init = rep(0.1,m); 
phi_init = rep(2, m);
p0_init = rep(0.3, m); 
factk = 0.75; gam0 = diag(m) * factk; gam0[upper.tri(gam0) | lower.tri(gam0)] = (1-factk)/(m-1);
gam_init = gam0[!c(diag(m))];
gamcov_init = rep(0.5, length(cov_names)*length(gam_init));
par0 = c(mu_init, phi_init, p0_init, gam_init, gamcov_init);
r3transcov <- optim(par = par0, fn = hmm_loglik_Transcovs, delta = rep(1,m)/m, m = m,y = d$icing.trans[d$wint1], 
                    covs = matrix(as.matrix(d[d$wint1, cov_names]), ncol = length(cov_names)), gr = NULL, method = "BFGS", control = list(maxit = 1000))
2 * (r3transcov$value + length(r3transcov$par)) #AIC = -818.3261
length(r3transcov$par) # = 27 = number of estimated parameters
predr3transcov_w2 = hmm_forecast_Transcovs(m = m, y = d$icing.trans[d$wint2], w_pars = r3transcov$par, delta = rep(1,m)/m, 
                                           covs = matrix(as.matrix(d[d$wint2, cov_names]), ncol = length(cov_names)), ahead = 12 )
predr3transcov_w3 = hmm_forecast_Transcovs(m = m, y = d$icing.trans[d$wint3], w_pars = r3transcov$par, delta = rep(1,m)/m, 
                                           covs = matrix(as.matrix(d[d$wint3, cov_names]), ncol = length(cov_names)), ahead = 12 )
predr3transcov_w2$CRPS_HMM #CRPS = 0.04725586
predr3transcov_w3$CRPS_HMM #CRPS = 0.04070712


mean(abs(predr3transcov_w2$Q2 - d$icing.trans[d$wint2]), na.rm = T) #MAE = 0.06152247
mean(abs(predr3transcov_w3$Q2 - d$icing.trans[d$wint3]), na.rm = T) #MAE = 0.05032206


save(r3transcov, predr3transcov_w2, predr3transcov_w3, file = "hmm3_transcov_models_preds.RData")



#4-state model
m = 4;
mu_init = rep(0.1,m); 
phi_init = rep(4, m);
p0_init = rep(0.3, m); 
factk = 0.75; gam0 = diag(m) * factk; gam0[upper.tri(gam0) | lower.tri(gam0)] = (1-factk)/(m-1);
gam_init = gam0[!c(diag(m))];
gamcov_init = rep(-.05, length(cov_names)*length(gam_init));
par0 = c(mu_init, phi_init, p0_init, gam_init, gamcov_init);
r4transcov <- optim(par = par0, fn = hmm_loglik_Transcovs, delta = rep(1,m)/m, m = m,y = d$icing.trans[d$wint1], 
                    covs = matrix(as.matrix(d[d$wint1, cov_names]), ncol = length(cov_names)), gr = NULL, method = "BFGS", control = list(maxit = 1000))
2 * (r4transcov$value + length(r4transcov$par)) #AIC = 450.933
predr4transcov_w2 = hmm_forecast_Transcovs(m = m, y = d$icing.trans[d$wint2], w_pars = r4transcov$par, delta = rep(1,m)/m, 
                                           covs = matrix(as.matrix(d[d$wint2, cov_names]), ncol = length(cov_names)), ahead = 12 )
predr4transcov_w3 = hmm_forecast_Transcovs(m = m, y = d$icing.trans[d$wint3], w_pars = r4transcov$par, delta = rep(1,m)/m, 
                                           covs = matrix(as.matrix(d[d$wint3, cov_names]), ncol = length(cov_names)), ahead = 12 )
predr4transcov_w2$CRPS_HMM #CRPS = 0.06365725
predr4transcov_w3$CRPS_HMM #CRPS = 0.05251342


#####################
### HMM dens covs ###
#####################
cov_names = c("cbh", "t");

#2-state model
m = 2;
mu_init = rep(0.1,m); 
phi_init = rep(4, m);
p0_init = rep(0.3, m); 
mucov_init = rep(0.01, length(cov_names)*m)
phicov_init = rep(0.01, length(cov_names)*m)
p0cov_init = rep(0.01, length(cov_names)*m)
factk = 0.75; gam0 = diag(m) * factk; gam0[upper.tri(gam0) | lower.tri(gam0)] = (1-factk)/(m-1);
gam_init = gam0[!c(diag(m))];
phi_covs = T; p0_covs = T;
par0 = c(mu_init, phi_init, p0_init, mucov_init, phicov_init, p0cov_init, gam_init);
#par0 = c(mu_init, phi_init, p0_init, mucov_init, gam_init);
#hmm_loglik_Denscovs(m = m, y =d$icing.trans[d$wint1], w_pars = par0,delta = rep(1,m)/m, covs = matrix(as.matrix(d[d$wint1, cov_names]), ncol = length(cov_names)), phi_covs = phi_covs, p0_covs = p0_covs )
r2denscov <- optim(par = par0, fn = hmm_loglik_Denscovs, m = m,y = d$icing.trans[d$wint1], delta = rep(1,m)/m,
                    covs = matrix(as.matrix(d[d$wint1, cov_names]), ncol = length(cov_names)), phi_covs = phi_covs, p0_covs = p0_covs, gr = NULL, method = "BFGS", control = list(maxit = 1000))
2 * (r2denscov$value + length(r2denscov$par)) #AIC = 
length(r2denscov$par) # = 20 = number of estimated parameters
predr2denscov_w2 = hmm_forecast_Denscov(m = m, y = d$icing.trans[d$wint2], w_pars = r2denscov$par, delta = rep(1,m)/m, 
                                           covs = matrix(as.matrix(d[d$wint2, cov_names]), ncol = length(cov_names)), ahead = 12, p0_covs = p0_covs )
predr2denscov_w3 = hmm_forecast_Denscov(m = m, y = d$icing.trans[d$wint3], w_pars = r2denscov$par, delta = rep(1,m)/m, 
                                           covs = matrix(as.matrix(d[d$wint3, cov_names]), ncol = length(cov_names)), ahead = 12, p0_covs = p0_covs )
predr2denscov_w2$CRPS_HMM #CRPS = 
predr2denscov_w3$CRPS_HMM #CRPS = 

mean(abs(predr2denscov_w2$Q2 - d$icing.trans[d$wint2]), na.rm = T) #MAE = 0.06152247
mean(abs(predr2denscov_w3$Q2 - d$icing.trans[d$wint3]), na.rm = T) #MAE = 0.05032206


#3-state model
m = 3;
mu_init = rep(0.2,m); 
phi_init = rep(4, m);
p0_init = rep(0.4, m); 
mucov_init = rep(0.01, length(cov_names)*m)
phicov_init = rep(0.01, length(cov_names)*m)
p0cov_init = rep(0.01, length(cov_names)*m)
factk = 0.75; gam0 = diag(m) * factk; gam0[upper.tri(gam0) | lower.tri(gam0)] = (1-factk)/(m-1);
gam_init = gam0[!c(diag(m))];
phi_covs = F; p0_covs = F;
par0 = c(mu_init, phi_init, p0_init, mucov_init, phicov_init, p0cov_init, gam_init);
par0 = c(mu_init, phi_init, p0_init, mucov_init, gam_init);
#hmm_loglik_Denscovs(m = m, y =d$icing.trans[d$wint1], w_pars = par0,delta = rep(1,m)/m, covs = matrix(as.matrix(d[d$wint1, cov_names]), ncol = length(cov_names)), phi_covs = phi_covs, p0_covs = p0_covs )
r3denscov <- optim(par = par0, fn = hmm_loglik_Denscovs, m = m,y = d$icing.trans[d$wint1], delta = rep(1,m)/m,
                   covs = matrix(as.matrix(d[d$wint1, cov_names]), ncol = length(cov_names)), phi_covs = phi_covs, p0_covs = p0_covs, gr = NULL, method = "BFGS", control = list(maxit = 1000))
2 * (r3denscov$value + length(r3denscov$par)) #AIC = 
predr3denscov_w2 = hmm_forecast_Denscov(m = m, y = d$icing.trans[d$wint2], w_pars = r3denscov$par, delta = rep(1,m)/m, 
                                        covs = matrix(as.matrix(d[d$wint2, cov_names]), ncol = length(cov_names)), ahead = 12, p0_covs = p0_covs )
predr3denscov_w3 = hmm_forecast_Denscov(m = m, y = d$icing.trans[d$wint3], w_pars = r3denscov$par, delta = rep(1,m)/m, 
                                        covs = matrix(as.matrix(d[d$wint3, cov_names]), ncol = length(cov_names)), ahead = 12, p0_covs = p0_covs )
predr3denscov_w2$CRPS_HMM #CRPS = 
predr3denscov_w3$CRPS_HMM #CRPS = 





rdens = optim(par = rep(1.1,14), fn = hmm_loglik_Denscovs, delta =c(0.5,0.5), m = 2,y = d$icing.trans[d$wint1], covs = as.matrix(as.matrix(d$t)[d$wint1, ]), gr = NULL, method = "BFGS", phi_covs = T, p0_covs = T )
2 * (rdens$value + length(rdens$par))





###################
### Forecasting ###
###################

predAR = AR1_forecast(y = d$icing.trans[d$wint2], ahead = 12, c_AR = ar.1$coef, sig2_AR = ar.1$sigma2 )
plot(d$icing[d$wint2], type = "l")
lines(predAR$Q1, col = 3)
lines(predAR$Q2, col = 4)
lines(predAR$Q3, col = 3)
lines(predAR$Mean, col = 2)
predAR$CRPS_AR
hist(predAR$pseudo[d$icing.trans[d$wint2] != 0])


#Checking crps calculation
y.val = d$icing.trans[d$wint2]
count = 0; foo = 0;
for(i in 1:length(predAR$Mean)){
  if(!is.na(y.val[i])){
    sim = rnorm(1000,predAR$Mean[i], sqrt(predAR$Variance[i]))
    foo = foo + crps(sim, y.val[i])  
    count = count + 1
  }
}

predHMM = hmm_forecast(m = 2, y = d$icing.trans[d$wint2], w_pars = r2$par, delta = rep(1)/2,ahead = 12, c_AR = ar.1$coef, sig2_AR = ar.1$sigma2 )
plot(d$icing[d$wint2], type = "l")
lines(predHMM$Q1, col = 3)
lines(predHMM$Q2, col = 4)
lines(predHMM$Q3, col = 3)
predHMM$mu
predHMM$phi
predHMM$p0
hist(predHMM$pseudo[d$icing.trans[d$wint2] != 0])
hist(predHMM$pseudo[d$icing.trans[d$wint2] == 0])


predHMM.transcov = hmm_forecast_Transcovs(m = 2, y = d$icing.trans[d$wint2], w_pars = r2cov$par, delta = rep(1)/2, covs = as.matrix(matrix(ts_dat)[d$wint2, ]),ahead = 12)
hist(predHMM.transcov$pseudo[d$icing.trans[d$wint2] != 0])
predHMM.transcov$CRPS_HMM



predHMM.denscov = hmm_forecast_Denscov(m = 2, y = d$icing.trans[d$wint2], w_pars = rdens$par, delta = rep(1)/2, covs = as.matrix(matrix(d$t)[d$wint2, ]),ahead = 12, phi_covs = T, p0_covs = T)
hist(predHMM.denscov$pseudo[d$icing.trans[d$wint2] != 0])
predHMM.denscov$CRPS_HMM








# //Conclusions/discussions:
#   
# // - logscore is not suitable evaluation for mixed distribution of continuous and discrete
# // - presented methodology is able to recover parameters from simulated data
# // - methodology to sample from predictive distribution
# // - another study could consider Kumaraswamy distribution instead of beta
# // - regression on trans probs vs. density pars
# // - analytical gradient information, find paper showing it incl. hessian
# // - make list of papers with similar methodology of beta hmm. Difference is the explicit forecast setting





####################################
###### FURTHER EXTRA ANALYSIS ###### 
####################################
###
### Compute correlation between daily average icing levels and daily number of missing observation
###
library(dplyr)
df = d
df$timestamp_col = d$X
# Convert the timestamp column to POSIXct format
df$timestamp_col <- as.POSIXct(df$timestamp_col, format = "%Y/%m/%d %H:%M:%S")
# Extract date and hour separately
df$date <- as.Date(df$timestamp_col)
df$hour <- format(df$timestamp_col, "%H")
# Group by date and calculate daily averages
daily_avg_df <- df %>%
  group_by(date) %>%
  summarize(daily_average_ice = mean(icing, na.rm = TRUE),
            daily_average_cbh = mean(cbh, na.rm = TRUE),
            daily_average_lnsp = mean(lnsp, na.rm = TRUE),
            daily_average_t = mean(t, na.rm = TRUE),
            daily_average_ws = mean(ws, na.rm = TRUE),
            daily_average_q = mean(q, na.rm = TRUE),
            daily_average_ciwc = mean(ciwc, na.rm = TRUE),
            daily_missing = sum(is.na(icing)))
# View the new dataframe
print(daily_avg_df)
cor(daily_avg_df$daily_missing, daily_avg_df$daily_average_ice, use = "pairwise.complete.obs")
cor(daily_avg_df$daily_missing, daily_avg_df$daily_average_cbh, use = "pairwise.complete.obs")
cor(daily_avg_df$daily_missing, daily_avg_df$daily_average_lnsp, use = "pairwise.complete.obs")
cor(daily_avg_df$daily_missing, daily_avg_df$daily_average_t, use = "pairwise.complete.obs")
cor(daily_avg_df$daily_missing, daily_avg_df$daily_average_ws, use = "pairwise.complete.obs")
cor(daily_avg_df$daily_missing, daily_avg_df$daily_average_q, use = "pairwise.complete.obs")
cor(daily_avg_df$daily_missing, daily_avg_df$daily_average_ciwc, use = "pairwise.complete.obs")



log.trans <- function(a,b,x){
  return( 1/(1 + a*exp(b*x)) ) 
}
log.trans(0.01, 0.015, 600)

