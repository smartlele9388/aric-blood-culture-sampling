# Actual implementation adapted from supplied fit_standardized_cif.R.
# Scientific specification preserved; exact covariate-profile aggregation and
# deterministic worker/checkpoint management accelerate the fixed 1000 attempts.
GROUPS <- c("low","moderate","high")
HORIZONS <- c(1,3,7)
ENDPOINTS <- c("sampling","bsi")
PREDICTORS <- c("baseline_group","age_per_10y","female","icu_group","admission_group")
prepare_data <- function(path) {
 d <- read.csv(path,stringsAsFactors=FALSE)
 req <- c("subject_id","stay_id",PREDICTORS,"bsi_time_days","bsi_status","sampling_time_days","sampling_status")
 stopifnot(all(req %in% names(d)),!anyNA(d[,req]),!anyDuplicated(d$stay_id),nrow(d)==13863,
           length(unique(d$subject_id))==12757,sum(d$bsi_status==1)==268,
           all(d$sampling_time_days>=0),all(d$bsi_time_days>0),
           all(d$sampling_time_days<=d$bsi_time_days+1e-10))
 d$baseline_group <- factor(d$baseline_group,levels=GROUPS)
 d$icu_group <- factor(d$icu_group)
 d$admission_group <- factor(d$admission_group)
 d
}
guard_value <- function(expr) {
 w <- character(); er <- NULL
 val <- tryCatch(withCallingHandlers(expr,warning=function(z) {
   w <<- c(w,conditionMessage(z));invokeRestart("muffleWarning")
 }),error=function(z) {er <<- conditionMessage(z);NULL})
 list(ok=is.null(er)&&length(w)==0,value=val,warnings=w,error=er)
}
check_predictions <- function(p,n) {
 stopifnot(is.matrix(p),identical(dim(p),c(as.integer(n),3L)),all(is.finite(p)),
  all(p>= -1e-10),all(p<=1+1e-10),all(p[,2]>=p[,1]-1e-12),all(p[,3]>=p[,2]-1e-12))
}
collapse_population <- function(d) {
 vars <- setdiff(PREDICTORS,"baseline_group")
 z <- data.table::as.data.table(d[,vars,drop=FALSE])
 z <- z[,list(weight=.N),by=vars]
 as.data.frame(z)
}
fit_endpoint <- function(d,endpoint,diagnostics=FALSE,validate_expanded=FALSE) {
 md <- d[,PREDICTORS,drop=FALSE]
 md$.time <- d[[paste0(endpoint,"_time_days")]]
 md$.status <- d[[paste0(endpoint,"_status")]]
 stopifnot(setequal(unique(md$.status[md$.status>0]),1:4))
 mm <- model.matrix(~baseline_group+age_per_10y+female+icu_group+admission_group,md)[,-1,drop=FALSE]
 stopifnot(qr(mm)$rank==ncol(mm))
 Hist <- prodlim::Hist
 fit <- riskRegression::CSC(Hist(.time,.status,cens.code=0)~baseline_group+age_per_10y+female+icu_group+admission_group,
  data=md,cause=1,surv.type="hazard",fitter="coxph",ties="efron",
  singular.ok=FALSE,na.action=na.fail,control=survival::coxph.control(timefix=FALSE,iter.max=50))
 stopifnot(inherits(fit,"CauseSpecificCox"),length(fit$models)==4,setequal(as.character(fit$causes),as.character(1:4)))
 for(m in fit$models) stopifnot(m$n==nrow(d),NROW(m$y)==nrow(d),all(is.finite(coef(m))),
   all(is.finite(m$var)),all(is.finite(m$loglik)),length(coef(m))==ncol(mm),
   max(m$iter)<50,qr(m$x)$rank==ncol(mm),min(eigen(m$var,symmetric=TRUE,only.values=TRUE)$values)>0)
 nd <- collapse_population(d)
 weights <- nd$weight/sum(nd$weight);nd$weight <- NULL
 risk_rows <- list();maxclosure <- 0;profile_error <- NA_real_
 for(g in GROUPS) {
  nd$baseline_group <- factor(rep(g,nrow(nd)),levels=GROUPS)
  total <- matrix(0,nrow(nd),3)
  for(cause in 1:4) {
   pp <- riskRegression::predictRisk(fit,newdata=nd,times=HORIZONS,cause=cause,
       product.limit=TRUE,diag=FALSE,iid=FALSE,average.iid=FALSE)
   check_predictions(pp,nrow(nd));total <- total+pp
   if(cause==1) p <- pp
  }
  sp <- predict(fit,newdata=nd,times=HORIZONS,cause=1,type="survival",product.limit=TRUE,
      diag=FALSE,se=FALSE,band=FALSE,iid=FALSE,confint=FALSE,average.iid=FALSE,
      keep.newdata=FALSE,keep.strata=FALSE)$survival
  check_predictions(1-sp,nrow(nd));cl <- max(abs(total+sp-1));stopifnot(cl<1e-8)
  maxclosure <- max(maxclosure,cl)
  stopifnot(all(HORIZONS<=max(md$.time)))
  meanrisk <- colSums(p*weights)
  if(validate_expanded) {
   expanded <- md[,PREDICTORS,drop=FALSE];expanded$baseline_group <- factor(rep(g,nrow(expanded)),levels=GROUPS)
   ep <- riskRegression::predictRisk(fit,newdata=expanded,times=HORIZONS,cause=1,
       product.limit=TRUE,diag=FALSE,iid=FALSE,average.iid=FALSE)
   check_predictions(ep,nrow(d));ee <- max(abs(colMeans(ep)-meanrisk))
   stopifnot(ee<1e-12);profile_error <- max(c(profile_error,ee),na.rm=TRUE)
  }
  risk_rows[[g]] <- data.frame(endpoint=endpoint,time_days=HORIZONS,group=g,estimate=100*meanrisk)
 }
 dg <- cf <- cv <- ph <- NULL
 if(diagnostics) {
  dg <- do.call(rbind,lapply(seq_along(fit$models),function(j) {
    m<-fit$models[[j]]
    data.frame(endpoint=endpoint,cause=fit$causes[j],n=m$n,events=m$nevent,
      parameters=length(coef(m)),matrix_rank=qr(m$x)$rank,iterations=max(m$iter),
      finite_coefficients=all(is.finite(coef(m))),finite_covariance=all(is.finite(m$var)),
      loglik_null=m$loglik[1],loglik_fitted=m$loglik[2],ties=m$method,
      minimum_covariance_eigenvalue=min(eigen(m$var,symmetric=TRUE,only.values=TRUE)$values),
      probability_closure_max_error=maxclosure,profile_collapse_max_error=profile_error)
  }))
  cf <- do.call(rbind,lapply(seq_along(fit$models),function(j) {
    m<-fit$models[[j]];b<-coef(m);se<-sqrt(diag(m$var))
    data.frame(endpoint=endpoint,cause=fit$causes[j],term=names(b),coefficient=unname(b),
      nonclustered_model_se=se,hazard_ratio=exp(b),nonclustered_ci_lower=exp(b-qnorm(.975)*se),nonclustered_ci_upper=exp(b+qnorm(.975)*se),
      nonclustered_p=2*pnorm(-abs(b/se)),purpose="fit diagnostics only; main CIF inference uses patient-cluster bootstrap")
  }))
  cv <- do.call(rbind,lapply(seq_along(fit$models),function(j) {
    m<-fit$models[[j]];v<-as.data.frame(as.table(m$var));names(v)<-c("row","col","covariance")
    v$row<-rep(names(coef(m)),times=length(coef(m)));v$col<-rep(names(coef(m)),each=length(coef(m)))
    data.frame(endpoint=endpoint,cause=fit$causes[j],v)
  }))
  ph <- do.call(rbind,lapply(seq_along(fit$models),function(j) {
    m<-fit$models[[j]];z<-survival::cox.zph(m,transform="km",terms=TRUE,global=TRUE)
    data.frame(endpoint=endpoint,cause=fit$causes[j],term=rownames(z$table),
      chisq=z$table[,1],df=z$table[,2],p=z$table[,3],transform="km")
  }))
 }
 list(risks=do.call(rbind,risk_rows),diagnostics=dg,coefficients=cf,covariance=cv,ph=ph,
   maxclosure=maxclosure,n_profiles=nrow(nd),fit=if(diagnostics)fit else NULL)
}
risk_differences <- function(risks) {
 out<-list();k<-0
 for(e in ENDPOINTS) for(h in HORIZONS) {
  z<-risks[risks$endpoint==e & risks$time_days==h,]; v<-setNames(z$estimate,z$group)
  for(g in c("moderate","high")) {k<-k+1;out[[k]]<-data.frame(endpoint=e,time_days=h,contrast=paste0(g,"_minus_low"),estimate=v[[g]]-v[["low"]])}
 }
 do.call(rbind,out)
}
bootstrap_one <- function(b) {
 set.seed(rep_seeds[b])
 start<-Sys.time();drawn<-sample.int(length(clusters),length(clusters),replace=TRUE)
 ix<-unlist(clusters[drawn],use.names=FALSE);bd<-d[ix,,drop=FALSE];rownames(bd)<-NULL
 rr<-lapply(ENDPOINTS,function(e)guard_value(fit_endpoint(bd,e)))
 status<-data.frame(replicate=b,success=all(vapply(rr,function(x)x$ok,logical(1))),
   n_resampled_stays=nrow(bd),n_selected_clusters=length(clusters),seed=rep_seeds[b],
   seconds=as.numeric(difftime(Sys.time(),start,units="secs")))
 issues<-list();k<-0
 for(j in seq_along(rr)) {
  if(length(rr[[j]]$warnings)>0)for(w in rr[[j]]$warnings){k<-k+1;issues[[k]]<-data.frame(replicate=b,endpoint=ENDPOINTS[j],type="warning_fatal",message=w)}
  if(!is.null(rr[[j]]$error)){k<-k+1;issues[[k]]<-data.frame(replicate=b,endpoint=ENDPOINTS[j],type="error",message=rr[[j]]$error)}
 }
 risks<-diffs<-NULL
 if(status$success) {
   risks<-do.call(rbind,lapply(rr,function(z)z$value$risks));risks$replicate<-b
   diffs<-risk_differences(risks);diffs$replicate<-b
   status$max_probability_closure_error<-max(vapply(rr,function(z)z$value$maxclosure,numeric(1)))
 }else status$max_probability_closure_error<-NA_real_
 ans<-list(status=status,risks=risks,diffs=diffs,issues=if(k)do.call(rbind,issues)else NULL)
 # Only aggregate estimates/status: this RDS contains no IDs, individual predictions or fits.
 target<-file.path(checkpoint_dir,sprintf("replicate_%04d.rds",b))
 temporary<-paste0(target,".writing")
 saveRDS(ans,temporary)
 if(!file.rename(temporary,target))stop("Unable to atomically finalize completed aggregate checkpoint.")
 ans
}
