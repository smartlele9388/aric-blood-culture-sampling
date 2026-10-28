#!/usr/bin/env Rscript
# Executed adaptation of the supplied template; details in template_changes.md.
# Usage: Rscript fit_standardized_cif.R task_root [B=1000] [workers=6] [pilot=3]
args<-commandArgs(TRUE);root<-normalizePath(args[1],winslash="/",mustWork=TRUE)
B<-if(length(args)>=2)as.integer(args[2])else 1000L
workers<-if(length(args)>=3)as.integer(args[3])else 6L
pilot<-if(length(args)>=4)as.integer(args[4])else 3L
stopifnot(B>=1000,workers>=1,pilot>=0)
.libPaths(c(file.path(root,"R_library"),.libPaths()))
suppressPackageStartupMessages({library(riskRegression);library(prodlim);library(survival);library(jsonlite);library(data.table)})
data.table::setDTthreads(1)
source(file.path(root,"code","standardization_functions.R"))
source(file.path(root,"code","standardization_lock_specification.R"))
input<-file.path(root,"secure_derived_inputs","cohort_standardization_S0.csv")
aggregate<-file.path(root,"aggregate");secure<-file.path(root,"secure_models")
dir.create(secure,showWarnings=FALSE);checkpoint_dir<-file.path(secure,"standardization_bootstrap_checkpoints")
dir.create(checkpoint_dir,showWarnings=FALSE)
lock_standardization(root)
write_csv<-function(x,file)write.csv(x,file.path(aggregate,file),row.names=FALSE,na="NA")
json<-function(x,file)jsonlite::write_json(x,file.path(aggregate,file),auto_unbox=TRUE,pretty=TRUE,digits=16,null="null")
writeLines(capture.output(sessionInfo()),file.path(root,"logs","standardization_sessionInfo.txt"))
bound<-jsonlite::read_json(file.path(aggregate,"standardization_api_tests.json"))
stopifnot(bound$zero$max_AJ_difference<1e-8,bound$ties$max_probability_closure_error<1e-8)
d<-prepare_data(input);clusters<-unname(split(seq_len(nrow(d)),d$subject_id))
seed<-20261015L;set.seed(seed);rep_seeds<-sample.int(.Machine$integer.max,B,replace=FALSE)
cfg<-list(B=B,master_seed=seed,input_md5=unname(tools::md5sum(input)),horizons=HORIZONS,
 formula="baseline_group + age_per_10y + female + icu_group + admission_group",n_stays=nrow(d),n_subjects=length(clusters))
configpath<-file.path(secure,"standardization_run_config.rds")
if(file.exists(configpath))stopifnot(identical(cfg,readRDS(configpath)))else saveRDS(cfg,configpath)
starttime<-Sys.time()
pointfile<-file.path(secure,"standardization_point_results.rds")
if(!file.exists(pointfile)) {
 cat("Point models started:",format(Sys.time()),"\n")
 pointfits<-lapply(ENDPOINTS,function(e){cat("Endpoint",e,"\n");guard_value(fit_endpoint(d,e,TRUE,TRUE))})
 issues<-do.call(rbind,lapply(seq_along(pointfits),function(j){z<-pointfits[[j]]
  msgs<-c(z$warnings,z$error);if(length(msgs))data.frame(endpoint=ENDPOINTS[j],message=msgs)else NULL}))
 if(!is.null(issues))write_csv(issues,"standardization_point_issues.csv")
 if(!all(vapply(pointfits,function(z)z$ok,logical(1))))stop("Point-model error or warning; see point issues.")
 pointfits<-lapply(pointfits,`[[`,"value");saveRDS(pointfits,pointfile)
}else pointfits<-readRDS(pointfile)
point<-do.call(rbind,lapply(pointfits,`[[`,"risks"));diffpoint<-risk_differences(point)
write_csv(point,"standardized_cif_point.csv");write_csv(diffpoint,"standardized_risk_differences_point.csv")
for(z in c("diagnostics","coefficients","covariance","ph")) {
 name<-switch(z,diagnostics="model_diagnostics.csv",coefficients="standardized_model_coefficients.csv",
              covariance="standardized_model_covariance.csv",ph="standardized_model_PH_diagnostics.csv")
 write_csv(do.call(rbind,lapply(pointfits,`[[`,z)),name)
}
cat("Point models complete:",format(Sys.time()),"\n")
cluster<-parallel::makePSOCKcluster(workers,outfile=file.path(root,"logs","standardization_workers.log"))
on.exit<-NULL
parallel::clusterExport(cluster,c("root","d","clusters","rep_seeds","checkpoint_dir"),envir=environment())
parallel::clusterEvalQ(cluster,{
 .libPaths(c(file.path(root,"R_library"),.libPaths()))
 suppressPackageStartupMessages({library(riskRegression);library(prodlim);library(survival);library(data.table)})
 data.table::setDTthreads(1);source(file.path(root,"code","standardization_functions.R"));NULL
})
if(pilot>0 && !file.exists(file.path(aggregate,"standardization_pilot_summary.json"))) {
 pilotdir<-file.path(secure,"standardization_pilot_checkpoints");dir.create(pilotdir,showWarnings=FALSE)
 parallel::clusterCall(cluster,function(path){checkpoint_dir<<-path;NULL},pilotdir)
 # Independent pilot seeds; pilot output never enters formal percentile intervals.
 formal_seeds<-rep_seeds;set.seed(seed+1L);rep_seeds<-sample.int(.Machine$integer.max,pilot)
 parallel::clusterExport(cluster,"rep_seeds",envir=environment())
 cat("Pilot",pilot,"attempts started:",format(Sys.time()),"\n")
 ps<-parallel::parLapply(cluster,seq_len(pilot),bootstrap_one)
 stat<-do.call(rbind,lapply(ps,`[[`,"status"));write_csv(stat,"standardization_pilot_attempts.csv")
 pi<-do.call(rbind,lapply(ps,`[[`,"issues"));if(!is.null(pi))write_csv(pi,"standardization_pilot_issues.csv")
 json(list(attempts=pilot,successes=sum(stat$success),seconds_per_replicate=stat$seconds,
     estimated_formal_seconds=mean(stat$seconds)*B/workers,formal_CI=FALSE),"standardization_pilot_summary.json")
 cat("Pilot complete; mean seconds",mean(stat$seconds),"estimated formal minutes",mean(stat$seconds)*B/workers/60,"\n")
 if(any(!stat$success))stop("Pilot numerical failures; inspect before formal execution.")
 rep_seeds<-formal_seeds;parallel::clusterExport(cluster,"rep_seeds",envir=environment())
 parallel::clusterCall(cluster,function(path){checkpoint_dir<<-path;NULL},checkpoint_dir)
}
existing<-which(file.exists(file.path(checkpoint_dir,sprintf("replicate_%04d.rds",seq_len(B)))))
todo<-setdiff(seq_len(B),existing);res<-vector("list",B)
if(length(existing))for(b in existing)res[[b]]<-readRDS(file.path(checkpoint_dir,sprintf("replicate_%04d.rds",b)))
cat("Formal bootstrap:",B,"planned attempts;",length(existing),"completed checkpoints;",workers,"workers\n")
chunks<-split(todo,ceiling(seq_along(todo)/(workers*2)))
for(batch in chunks) {
 z<-parallel::parLapply(cluster,batch,bootstrap_one)
 for(i in seq_along(batch))res[[batch[i]]]<-z[[i]]
 stat<-do.call(rbind,lapply(res[!vapply(res,is.null,logical(1))],`[[`,"status"))
 write_csv(stat,"bootstrap_attempt_status.csv")
 cat(format(Sys.time()),"attempted",nrow(stat),"/",B,"success",sum(stat$success),"\n");flush.console()
}
parallel::stopCluster(cluster)
status<-do.call(rbind,lapply(res,`[[`,"status"));issues<-do.call(rbind,lapply(res,`[[`,"issues"))
if(is.null(issues))issues<-data.frame(replicate=integer(),endpoint=character(),type=character(),message=character())
write_csv(status,"bootstrap_attempt_status.csv");write_csv(issues,"bootstrap_warnings_and_failures.csv")
rr<-do.call(rbind,lapply(res,`[[`,"risks"));rd<-do.call(rbind,lapply(res,`[[`,"diffs"))
write_csv(rr,"bootstrap_standardized_cif.csv");write_csv(rd,"bootstrap_standardized_differences.csv")
success<-sum(status$success);formalok<-success/B>=.95
intervals<-function(pt,boot,key) {
 out<-pt;out$ci_lower<-out$ci_upper<-NA_real_
 for(i in seq_len(nrow(pt))) {
  q<-boot$estimate[boot$endpoint==pt$endpoint[i]&boot$time_days==pt$time_days[i]&boot[[key]]==pt[[key]][i]]
  stopifnot(length(q)==success,all(is.finite(q)))
  ci<-quantile(q,c(.025,.975),type=7,names=FALSE);out$ci_lower[i]<-ci[1];out$ci_upper[i]<-ci[2]
 }
 out$successful_replicates<-success;out$attempted_replicates<-B
 out$interval_method<-"patient-cluster percentile bootstrap; paired endpoints and contrasts; empirical resampled admission population"
 out$status<-if(formalok)"completed_exploratory_model_estimates"else"unstable_failure_rate_above_5_percent"
 out
}
if(success>0) {
 write_csv(intervals(point,rr,"group"),"standardized_cif.csv")
 write_csv(intervals(diffpoint,rd,"contrast"),"standardized_risk_differences.csv")
}
json(list(status=if(formalok)"completed"else"unstable",seed=seed,planned_attempts=B,completed_attempts=nrow(status),
 successes=success,failures=B-success,success_fraction=success/B,pilot_attempts=pilot,
 execution_mode=if(length(existing)==B)"export_only_resume_from_all_completed_attempts"else"fit_unfinished_attempts_and_export",
 previously_completed_attempts_reused=length(existing),new_formal_attempts_in_this_invocation=length(todo),
 workers=workers,n_subject_clusters=length(clusters),n_original_stays=nrow(d),
 point_fit_seconds_in_this_invocation=NA,run_started=format(starttime),run_completed=format(Sys.time()),
 current_invocation_seconds=as.numeric(difftime(Sys.time(),starttime,units="secs")),
 replicate_seed_generation="set.seed(20261015); sample.int(.Machine$integer.max,1000,replace=FALSE); each replicate set.seed(its saved seed)",
 all_repeatedly_selected_patient_admissions_retained=TRUE,standard_population="each replicate's own resampled admissions, equal admission weights",
 all_causes_modeled=1:4,probability_closure_max_error=max(status$max_probability_closure_error,na.rm=TRUE),
 zero_times_preserved=TRUE,no_failed_attempt_replacement=TRUE,
 inference="Bootstrap interval is descriptive and model-dependent; PH departures do not disappear through resampling."),"bootstrap_execution_summary.json")
cat("FORMAL STANDARDIZATION COMPLETE",format(Sys.time()),"successes",success,"/",B,"\n")
