args<-commandArgs(TRUE);root<-args[1];phase<-if(length(args)>=2)args[2]else"resume";lib<-file.path(root,"R_library")
stopifnot(phase%in%c("resume","final"))
.libPaths(c(lib,.libPaths()))
path<-file.path(root,"secure_models","standardization_bootstrap_checkpoints")
files<-list.files(path,pattern="^replicate_[0-9]+[.]rds$",full.names=TRUE)
rows<-lapply(files,function(f) {
 z<-readRDS(f);b<-as.integer(sub("replicate_([0-9]+)[.]rds","\\1",basename(f)))
 stopifnot(is.list(z),identical(names(z),c("status","risks","diffs","issues")),
  nrow(z$status)==1,z$status$replicate==b,length(z$status$success)==1,!is.na(z$status$success))
 if(z$status$success)stopifnot(nrow(z$risks)==18,nrow(z$diffs)==12,all(is.finite(z$risks$estimate)),all(is.finite(z$diffs$estimate)))
 z$status
})
s<-do.call(rbind,rows);stopifnot(!anyDuplicated(s$replicate))
if(phase=="final")stopifnot(nrow(s)==1000,setequal(s$replicate,1:1000))
write.csv(s,file.path(root,"aggregate",paste0("bootstrap_checkpoint_",phase,"_audit.csv")),row.names=FALSE)
jsonlite::write_json(list(checkpoints=length(files),all_readable=TRUE,all_complete=TRUE,
  replicate_ids=s$replicate,successful=sum(s$success),failed=sum(!s$success),
  next_unfinished=setdiff(1:1000,s$replicate),audit_time=format(Sys.time())),
  file.path(root,"logs",paste0("standardization_checkpoint_",phase,"_audit.json")),pretty=TRUE,auto_unbox=TRUE)
cat(length(files),"complete valid checkpoints; IDs unique; failures retained.\n")
