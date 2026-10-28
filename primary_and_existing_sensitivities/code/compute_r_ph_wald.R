# Joint patient-cluster robust Wald checks for prespecified time interactions.
# These are exploratory diagnostics using interval-start time transformations.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript compute_r_ph_wald.R r_batch_results_directory")
batch <- args[[1L]]
models <- list(
  day7 = list(model = "day7_time_interactions_final_r",
              terms = c("vmod:late7", "late7:vhigh", "exposure_indication_present:late7"),
              transform = "step at interval start day 7"),
  logtime = list(model = "logtime_main_interactions_final_r",
                 terms = c("vmod:log_t", "log_t:vhigh", "exposure_indication_present:log_t"),
                 transform = "log(1 + interval start day)")
)
rows <- list()
for (name in names(models)) {
  spec <- models[[name]]
  b <- read.csv(file.path(batch, paste0(spec$model, "_coefficients.csv")), check.names = FALSE)
  v <- as.matrix(read.csv(file.path(batch, paste0(spec$model, "_covariance.csv")),
                          row.names = 1, check.names = FALSE))
  if (!all(spec$terms %in% b$term) || !all(spec$terms %in% colnames(v)))
    stop("Missing time interaction term in ", spec$model, ": ", paste(setdiff(spec$terms, b$term), collapse = ","))
  beta <- b$coefficient[match(spec$terms, b$term)]
  subcov <- v[spec$terms, spec$terms, drop = FALSE]
  wald <- as.numeric(t(beta) %*% solve(subcov, beta))
  rows[[name]] <- data.frame(model = spec$model, block = "main score contrasts and indication",
                            time_transform = spec$transform, terms = paste(spec$terms, collapse = " | "),
                            chi_square_patient_cluster_robust_Wald = wald, df = length(beta),
                            p_value_exploratory = pchisq(wald, df = length(beta), lower.tail = FALSE),
                            model_intervals = unique(b$intervals), model_events = unique(b$events),
                            note = "Null p value does not establish PH; interactions use interval-start time",
                            stringsAsFactors = FALSE)
}
output <- do.call(rbind, rows)
write.csv(output, file.path(dirname(batch), "r_cluster_robust_main_ph_wald.csv"), row.names = FALSE)
print(output[, c("model", "chi_square_patient_cluster_robust_Wald", "df", "p_value_exploratory")])
