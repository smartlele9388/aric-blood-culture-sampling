args <- commandArgs(TRUE)
root <- args[1]
.libPaths(c(file.path(root,"R_library"),.libPaths()))
suppressPackageStartupMessages({library(riskRegression); library(prodlim); library(survival)})
sink(file.path(root,"logs","standardization_api_tests.log"),split=TRUE)
print(sessionInfo())
print(formals(riskRegression::CSC))
print(formals(riskRegression:::predict.CauseSpecificCox))
print(formals(riskRegression:::predictRisk.CauseSpecificCox))
aj <- function(time,status,horizons) {
  surv <- 1; ci <- rep(0,4); ans <- matrix(0,length(horizons),4)
  for(j in seq_along(horizons)) {
    surv <- 1; ci <- rep(0,4)
    for(t in sort(unique(time[time <= horizons[j]]))) {
      risk <- sum(time >= t); dn <- tabulate(status[time==t]+1,5)[2:5]
      ci <- ci+surv*dn/risk; surv <- surv*(1-sum(dn)/risk)
    }
    ans[j,] <- ci
  };ans
}
run_toy <- function(toy, label) {
 f <- CSC(Hist(time,status,cens.code=0)~1,data=toy,cause=1,
  surv.type="hazard",fitter="coxph",ties="efron",singular.ok=FALSE,
  na.action=na.fail,control=coxph.control(timefix=FALSE))
 h <- c(0,1,3)
 ps <- sapply(1:4,function(k) as.numeric(predictRisk(f,newdata=toy[1,,drop=FALSE],times=h,cause=k,
   product.limit=TRUE,diag=FALSE,iid=FALSE,average.iid=FALSE)))
 ss <- predict(f,newdata=toy[1,,drop=FALSE],times=h,cause=1,type="survival",product.limit=TRUE,
    diag=FALSE,se=FALSE,band=FALSE,iid=FALSE,confint=FALSE,average.iid=FALSE,
    keep.newdata=FALSE,keep.strata=FALSE)$survival
 aa <- aj(toy$time,toy$status,h)
 print(list(test=label, csc=ps,AJ=aa,survival=ss,maxdiff=max(abs(ps-aa)),closure=max(abs(rowSums(ps)+as.numeric(ss)-1))))
 list(label=label,max_AJ_difference=max(abs(ps-aa)),max_probability_closure_error=max(abs(rowSums(ps)+as.numeric(ss)-1)))
}
toy0 <- data.frame(time=c(0,1,2,3,4,5,8,10), status=c(1,2,1,2,3,4,0,0))
a <- run_toy(toy0,"zero_time_no_ties")
stopifnot(a$max_AJ_difference < 1e-8,a$max_probability_closure_error<1e-8)
toy1 <- data.frame(time=c(0,0,1,1,2,3,3,4,5,8,10,10),status=c(1,1,1,2,3,1,4,2,3,4,0,0))
b <- run_toy(toy1,"zero_time_and_tied_multiple_causes")
stopifnot(b$max_probability_closure_error < 1e-8)
clusters <- list(c(1L,2L),3L,4L)
ix <- unlist(clusters[c(1L,1L,3L)],use.names=FALSE)
stopifnot(identical(ix,c(1L,2L,1L,2L,4L)))
jsonlite::write_json(list(zero=a,ties=b,repeated_patient_all_admissions_preserved=TRUE,
 note="Tied Efron Cox baseline hazards may differ from nonparametric AJ; quantified, not forced equal."),
 file.path(root,"aggregate","standardization_api_tests.json"),pretty=TRUE,auto_unbox=TRUE,digits=16)
cat("ALL BOUNDARY TESTS PASSED\n")
sink()
