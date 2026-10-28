# Two patient-clustered S0 counting-process Cox fits in one fixed era-classifiable
# cohort. Preserve original main settings; broad_period is the only model delta.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=1L) stop("Usage: Rscript period_fit_cox.R OUTPUT_ROOT")
root <- normalizePath(args[[1]],winslash="/",mustWork=TRUE)
out <- file.path(root,"aggregate")
library(survival)
risk <- read.csv(gzfile(file.path(root,"secure_derived_inputs","period_matched_risk_S0.csv.gz")),stringsAsFactors=FALSE,check.names=FALSE)
stopifnot(length(unique(risk$stay_id))==12064L,sum(risk$event)==233L,all(risk$t_stop>risk$t_start),all(is.finite(risk$t_stop)))
risk$exposure_vulnerability_category <- factor(risk$exposure_vulnerability_category,levels=c("low","moderate","high"))
risk$icu_group <- factor(risk$icu_group)
risk$admission_group <- factor(risk$admission_group)
risk$cvc_type_group <- factor(risk$cvc_type_group)
risk$broad_period <- factor(risk$broad_period,levels=c("2008-2013","2014-2019","2020-2022"))
base_formula <- Surv(t_start,t_stop,event) ~ exposure_vulnerability_category +
  exposure_indication_present + age_per_10y + female + icu_group +
  admission_group + multiple_line + cvc_type_group +
  exposure_antibiotic_exposure + exposure_lactate_available + exposure_aki_assessable
fit_one <- function(formula,name) {
  started <- format(Sys.time(),"%Y-%m-%dT%H:%M:%S%z")
  warnings <- character(0)
  fit <- withCallingHandlers(coxph(formula,data=risk,ties="efron",robust=TRUE,
     cluster=subject_id,singular.ok=FALSE,x=TRUE,
     control=coxph.control(eps=1e-10,iter.max=50,timefix=FALSE)),
     warning=function(w){warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
  beta <- as.numeric(coef(fit)); terms <- names(coef(fit)); variance <- vcov(fit)
  se <- sqrt(diag(variance))
  stopifnot(fit$n==nrow(risk),fit$nevent==sum(risk$event),all(is.finite(beta)),
    all(is.finite(variance)),all(se>0),max(abs(beta))<15,qr(fit$x)$rank==length(beta),
    qr(variance)$rank==length(beta),max(fit$iter)<50)
  tab <- data.frame(model=name,term=terms,coefficient=beta,robust_se=se,
    hazard_ratio=exp(beta),ci_lower=exp(beta-qnorm(.975)*se),ci_upper=exp(beta+qnorm(.975)*se),
    p_value=2*pnorm(-abs(beta/se)),intervals=nrow(risk),stays=length(unique(risk$stay_id)),
    patients=length(unique(risk$subject_id)),events=sum(risk$event))
  write.csv(tab,file.path(out,paste0(name,"_coefficients.csv")),row.names=FALSE)
  write.csv(variance,file.path(out,paste0(name,"_covariance.csv")),row.names=TRUE)
  naive <- fit$naive.var;dimnames(naive)<-list(terms,terms)
  write.csv(naive,file.path(out,paste0(name,"_naive_covariance.csv")),row.names=TRUE)
  diag <- data.frame(model=name,status="completed",started_at=started,
    completed_at=format(Sys.time(),"%Y-%m-%dT%H:%M:%S%z"),survival_version=as.character(packageVersion("survival")),
    r_version=R.version.string,intervals=nrow(risk),events=sum(risk$event),
    patients=length(unique(risk$subject_id)),stays=length(unique(risk$stay_id)),risk_days=sum(risk$duration_days),
    fitted_rows=fit$n,fitted_events=fit$nevent,loglik=fit$loglik[[2]],iterations=fit$iter,
    parameters=length(beta),design_rank=qr(fit$x)$rank,covariance_rank=qr(variance)$rank,
    robust_variance_min_eigenvalue=min(eigen(variance,symmetric=TRUE,only.values=TRUE)$values),
    model_formula=paste(deparse(formula),collapse=" "),ties="efron",cluster="subject_id",timefix=FALSE,
    finite_coefficients=all(is.finite(beta)),finite_covariance=all(is.finite(variance)),
    warnings=paste(warnings,collapse=" | "),stringsAsFactors=FALSE)
  write.csv(diag,file.path(out,paste0(name,"_diagnostics.csv")),row.names=FALSE)
  zph <- cox.zph(fit,transform=function(t)log1p(t),terms=TRUE,global=TRUE)
  z <- data.frame(term=rownames(zph$table),zph$table,row.names=NULL,check.names=FALSE)
  write.csv(z,file.path(out,paste0(name,"_zph_log_event_time.csv")),row.names=FALSE)
  cat(name,":",nrow(risk),"intervals;",sum(risk$event),"events;",length(beta),"terms; warnings:",paste(warnings,collapse=" | "),"\n")
  fit$x
}
x0 <- fit_one(base_formula,"period_cox_without_era")
x1 <- fit_one(update(base_formula,~ . + broad_period),"period_cox_with_era")
stopifnot(all(colnames(x0)%in%colnames(x1)),
  identical(dim(x0),dim(x1[,colnames(x0),drop=FALSE])),
  identical(as.vector(x0),as.vector(x1[,colnames(x0),drop=FALSE])))
write.csv(data.frame(same_rows=nrow(risk),shared_columns=ncol(x0),shared_design_identical=TRUE,
  maximum_numeric_difference=max(abs(x0-x1[,colnames(x0),drop=FALSE])),
  added_columns=paste(setdiff(colnames(x1),colnames(x0)),collapse=" | ")),
  file.path(out,"period_cox_matched_design_check.csv"),row.names=FALSE)
