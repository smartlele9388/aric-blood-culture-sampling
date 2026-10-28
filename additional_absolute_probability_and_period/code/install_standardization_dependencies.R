# Run only in the new analysis directory; never writes original R libraries.
# Rscript install_standardization_dependencies.R task_root
args<-commandArgs(TRUE);stopifnot(length(args)==1)
root<-normalizePath(args[1],winslash="/",mustWork=TRUE)
local_library<-file.path(root,"R_library")
dir.create(local_library,showWarnings=FALSE,recursive=TRUE)
.libPaths(c(local_library,.libPaths()));options(timeout=300)
needed<-c("riskRegression","prodlim","jsonlite","data.table","digest")
missing<-needed[!vapply(needed,requireNamespace,logical(1),quietly=TRUE)]
if(length(missing))install.packages(missing,lib=local_library,repos="https://cloud.r-project.org",
 type=if(.Platform$OS.type=="windows")"binary"else"source")
stopifnot(all(vapply(needed,requireNamespace,logical(1),quietly=TRUE)))
versions<-data.frame(package=c("R",needed,"survival"),version=c(as.character(getRversion()),
 vapply(c(needed,"survival"),function(p)as.character(packageVersion(p)),character(1))))
print(versions)
write.csv(versions,file.path(root,"logs","standardization_installed_versions.csv"),row.names=FALSE)
cat("Compare versions against the delivered standardization sessionInfo and scientific lock before any reproduction.\n")
cat("If repositories now deliver newer versions, rerun all API/boundary tests and use a fresh output directory; do not reuse checkpoints from a different lock.\n")
