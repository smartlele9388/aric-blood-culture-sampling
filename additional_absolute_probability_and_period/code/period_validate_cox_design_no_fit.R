# Recover only the post-fit design validation. Does not fit or refit a model.
args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=1L)stop("Usage: Rscript period_validate_cox_design_no_fit.R OUTPUT_ROOT")
root<-normalizePath(args[[1]],winslash="/",mustWork=TRUE)
risk<-read.csv(gzfile(file.path(root,"secure_derived_inputs","period_matched_risk_S0.csv.gz")),stringsAsFactors=FALSE,check.names=FALSE)
risk$exposure_vulnerability_category<-factor(risk$exposure_vulnerability_category,levels=c("low","moderate","high"))
risk$icu_group<-factor(risk$icu_group)
risk$admission_group<-factor(risk$admission_group)
risk$cvc_type_group<-factor(risk$cvc_type_group)
risk$broad_period<-factor(risk$broad_period,levels=c("2008-2013","2014-2019","2020-2022"))
base<- ~ exposure_vulnerability_category + exposure_indication_present + age_per_10y + female +
  icu_group + admission_group + multiple_line + cvc_type_group + exposure_antibiotic_exposure +
  exposure_lactate_available + exposure_aki_assessable
x0<-model.matrix(base,risk)
x1<-model.matrix(update(base,~ . + broad_period),risk)
same<-x1[,colnames(x0),drop=FALSE]
stopifnot(identical(dim(x0),dim(same)),identical(as.vector(x0),as.vector(same)))
cat("Independent no-fit reconstruction: all",nrow(x0),"rows and",ncol(x0)-1,
  "shared Cox predictor columns are numerically identical. Maximum difference:",max(abs(x0-same)),"\n")
cat("Attribute comparison of original-style matrices:",identical(unname(x0),unname(same)),
  "; numeric equality:",identical(as.vector(x0),as.vector(same)),"\n")
write.csv(data.frame(same_rows=nrow(risk),shared_columns=ncol(x0)-1L,shared_design_identical=TRUE,
  maximum_numeric_difference=max(abs(x0-same)),original_failure="identical() compared matrix attributes, not just model values",
  repair="validation only; model.matrix reconstructed, no refit",added_columns=paste(setdiff(colnames(x1),colnames(x0)),collapse=" | ")),
  file.path(root,"aggregate","period_cox_matched_design_check.csv"),row.names=FALSE)
