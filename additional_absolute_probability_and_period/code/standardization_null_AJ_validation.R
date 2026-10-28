# Numerical implementation check, not a competing scientific model comparison.
args<-commandArgs(TRUE);root<-args[1];.libPaths(c(file.path(root,"R_library"),.libPaths()))
suppressPackageStartupMessages({library(riskRegression);library(prodlim);library(survival)})
d<-read.csv(file.path(root,"secure_derived_inputs","cohort_standardization_S0.csv"))
h<-c(0,1,3,7);out<-list();k<-0
for(e in c("sampling","bsi")) {
 md<-data.frame(time=d[[paste0(e,"_time_days")]],status=d[[paste0(e,"_status")]])
 stopifnot(!any(md$status==0 & md$time<28-1e-10))
 f<-CSC(Hist(time,status,cens.code=0)~1,data=md,cause=1,surv.type="hazard",fitter="coxph",ties="efron",
   singular.ok=FALSE,na.action=na.fail,control=coxph.control(timefix=FALSE))
 pp<-as.numeric(predictRisk(f,newdata=md[1,,drop=FALSE],times=h,cause=1,product.limit=TRUE))
 # Independent nonparametric implementation: explicit risk sets/event-before-censor rule.
 aa<-vapply(h,function(tt) {
  surv<-1;ci<-0
  for(t in sort(unique(md$time[md$time<=tt]))) {
   r<-sum(md$time>=t);ne<-sum(md$time==t & md$status==1);all<-sum(md$time==t & md$status>0)
   ci<-ci+surv*ne/r;surv<-surv*(1-all/r)
  };ci
 },numeric(1))
 frac<-vapply(h,function(tt)mean(md$time<=tt & md$status==1),numeric(1))
 stopifnot(max(abs(aa-frac))<1e-12)
 for(i in seq_along(h)) {k<-k+1;out[[k]]<-data.frame(endpoint=e,time_days=h[i],
   independent_AJ=aa[i],event_fraction_no_early_censoring=frac[i],null_CSC_efron_product_limit=pp[i],
   csc_minus_AJ=pp[i]-aa[i],purpose="numerical validation only; not an added scientific comparison")}
}
res<-do.call(rbind,out)
write.csv(res,file.path(root,"aggregate","standardization_null_AJ_validation.csv"),row.names=FALSE)
print(res);cat("Real-data no-covariate validation complete; tied Efron vs AJ differences quantified.\n")
