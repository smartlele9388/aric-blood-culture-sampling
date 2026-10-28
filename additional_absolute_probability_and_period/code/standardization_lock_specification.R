# Invocation before/after orchestration changes. Lock only statistical functions
# plus exact package versions; worker count and atomic-save code are orchestration.
lock_standardization <- function(root) {
  if(!requireNamespace("digest",quietly=TRUE))stop("digest dependency missing")
  functions<-c("fit_endpoint","collapse_population","check_predictions","risk_differences","prepare_data")
  scientific_functions<-setNames(lapply(functions,function(nm)paste(deparse(get(nm)),collapse="\n")),functions)
  spec<-list(functions_sha256=digest::digest(scientific_functions,algo="sha256"),
    packages=setNames(as.list(vapply(c("riskRegression","prodlim","survival","data.table"),function(p)as.character(packageVersion(p)),character(1))),c("riskRegression","prodlim","survival","data.table")),
    r_version=as.character(getRversion()),master_seed=20261015L,groups=GROUPS,horizons=HORIZONS,
    seed_schedule="sample.int(.Machine$integer.max,1000,replace=FALSE) after set.seed(20261015)",
    bootstrap_target="equal resampled admission weights; duplicate patient draws retain all admissions",
    ties="efron",timefix=FALSE)
  target<-file.path(root,"secure_models","standardization_scientific_lock.rds")
  if(file.exists(target))stopifnot(identical(spec,readRDS(target)))else saveRDS(spec,target)
  jsonlite::write_json(spec,file.path(root,"aggregate","standardization_scientific_lock.json"),auto_unbox=TRUE,pretty=TRUE,digits=16)
  invisible(spec)
}
if(sys.nframe()==0L) {
 root<-commandArgs(TRUE)[1];.libPaths(c(file.path(root,"R_library"),.libPaths()))
 source(file.path(root,"code","standardization_functions.R"));lock_standardization(root)
 cat("Scientific function fingerprint and package versions locked.\n")
}
