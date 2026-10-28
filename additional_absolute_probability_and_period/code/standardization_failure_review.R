# Reconstruct only failed replicate membership from its saved seed, without fitting
# another model. Export no IDs or patient rows; rare positive counts are suppressed.
args<-commandArgs(TRUE);root<-args[1];.libPaths(c(file.path(root,"R_library"),.libPaths()))
source(file.path(root,"code","standardization_functions.R"));d<-prepare_data(file.path(root,"secure_derived_inputs","cohort_standardization_S0.csv"))
clusters<-unname(split(seq_len(nrow(d)),d$subject_id));s<-read.csv(file.path(root,"aggregate","bootstrap_attempt_status.csv"))
stopifnot(nrow(s)==1000,!anyDuplicated(s$replicate),setequal(s$replicate,1:1000))
bad<-s[!s$success,,drop=FALSE];ans<-list();k<-0
for(i in seq_len(nrow(bad))) {
 b<-bad$replicate[i];set.seed(bad$seed[i]);ix<-unlist(clusters[sample.int(length(clusters),length(clusters),replace=TRUE)],use.names=FALSE)
 z<-d[ix,,drop=FALSE];stopifnot(nrow(z)==bad$n_resampled_stays[i])
 for(e in ENDPOINTS)for(v in c("baseline_group","icu_group","admission_group","female"))for(g in unique(as.character(d[[v]])))for(cause in 1:4) {
  n<-sum(as.character(z[[v]])==g & z[[paste0(e,"_status")]]==cause)
  if(n<11) {k<-k+1;ans[[k]]<-data.frame(replicate=b,endpoint=e,variable=v,level=as.character(g),cause=cause,
    event_support=if(n==0)"zero"else"1-10",interpretation="candidate sparse-support explanation; original numerical warning remains the failure criterion")}
 }
}
out<-if(k)do.call(rbind,ans)else data.frame(replicate=integer(),endpoint=character(),variable=character(),level=character(),cause=integer(),event_support=character(),interpretation=character())
write.csv(out,file.path(root,"aggregate","bootstrap_failed_replicate_support_review.csv"),row.names=FALSE)
cat("Failed complete attempts reviewed without any refit:",nrow(bad),"\n")
if(nrow(out))print(out[out$event_support=="zero",])
