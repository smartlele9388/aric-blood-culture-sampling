# Generic, reviewer-runnable Cox refit. Manifest rows specify local derived-risk
# inputs and R right-hand sides; this script writes model-level aggregates only.
# Usage: Rscript run_cox_batch_survival.R manifest.tsv output_directory
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript run_cox_batch_survival.R manifest.tsv output_directory")
manifest_path <- normalizePath(args[[1L]], winslash = "/", mustWork = TRUE)
output_dir <- args[[2L]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
library(survival)
manifest <- read.delim(manifest_path, stringsAsFactors = FALSE, check.names = FALSE,
                       na.strings = character())
needed <- c("model_name", "input_path", "rhs", "score_delta_path", "transform")
if (!all(needed %in% names(manifest))) stop("Manifest requires: ", paste(needed, collapse = ", "))
if (anyDuplicated(manifest$model_name)) stop("Model names must be unique")
if (any(grepl("[^A-Za-z0-9_-]", manifest$model_name))) stop("Invalid model name")
cache <- new.env(parent = emptyenv())

read_frame <- function(path) {
  if (!file.exists(path)) stop("Missing local input: ", path)
  if (!exists(path, envir = cache, inherits = FALSE)) {
    value <- if (grepl("\\.gz$", path)) read.csv(gzfile(path), check.names = FALSE) else read.csv(path, check.names = FALSE)
    assign(path, value, envir = cache)
  }
  get(path, envir = cache, inherits = FALSE)
}

apply_score_delta <- function(data, path) {
  if (is.na(path) || !nzchar(path)) return(data)
  delta <- read_frame(path)
  needed_delta <- c("stay_id", "delta_cancer", "delta_meta", "delta_score")
  if (!all(needed_delta %in% names(delta))) stop("Incomplete score delta")
  if (anyDuplicated(delta$stay_id)) stop("Duplicated stay in score delta")
  index <- match(data$stay_id, delta$stay_id)
  if (anyNA(index)) stop("Risk stay not found in score delta")
  if (!all(delta$delta_cancer + delta$delta_meta == delta$delta_score)) stop("Invalid score delta")
  for (name in c("cancer", "metastatic_cancer", "corrected_chronic_score", "lag_corrected_score",
                 "corrected_score", "exposure_vulnerability_burden_score", "documented_chronic_count")) {
    if (name %in% names(data)) {
      add <- if (name == "cancer") delta$delta_cancer[index] else
        if (name == "metastatic_cancer") delta$delta_meta[index] else delta$delta_score[index]
      data[[name]] <- data[[name]] + add
    }
  }
  if ("exposure_vulnerability_category" %in% names(data)) {
    score <- if ("exposure_vulnerability_burden_score" %in% names(data))
      data$exposure_vulnerability_burden_score else data$lag_corrected_score
    data$exposure_vulnerability_category <- ifelse(score <= 1, "low", ifelse(score <= 3, "moderate", "high"))
    if ("vmod" %in% names(data)) data$vmod <- as.integer(data$exposure_vulnerability_category == "moderate")
    if ("vhigh" %in% names(data)) data$vhigh <- as.integer(data$exposure_vulnerability_category == "high")
  }
  data
}

transform_frame <- function(data, transform) {
  if (is.na(transform) || !nzchar(transform) || transform == "none") return(data)
  if (transform == "record_frequency_indication") {
    data$exposure_indication_present <- as.integer(data$lag_original_indication_with_frequency)
  } else if (transform == "without_creatinine_marker") {
    score <- data$lag_corrected_score - as.integer(data$lag_aki == 1 & !is.na(data$lag_aki))
    data$exposure_vulnerability_category <- ifelse(score <= 1, "low", ifelse(score <= 3, "moderate", "high"))
  } else if (transform == "nonoverlapping_cancer") {
    score <- data$lag_corrected_score - as.integer(data$cancer == 1 & data$metastatic_cancer == 1)
    data$exposure_vulnerability_category <- ifelse(score <= 1, "low", ifelse(score <= 3, "moderate", "high"))
  } else if (transform == "both_labs_observed") {
    data <- data[data$exposure_lactate_available == 1 & data$exposure_aki_assessable == 1, , drop = FALSE]
    data$icu_group[data$icu_group == "neuro"] <- "surgical_trauma"
  } else if (transform == "recognized_pathogen") {
    data$event <- data$event * as.integer(!is.na(data$recognized_pathogen) & data$recognized_pathogen == 1)
  } else if (transform == "logtime_main") {
    data$log_t <- log1p(data$t_start)
  } else if (transform %in% c("ph_lactate_time", "ph_aki_time", "ph_labs_time")) {
    log_t <- log1p(data$t_start)
    if (transform %in% c("ph_lactate_time", "ph_labs_time"))
      data$ph_lactate_available <- data$exposure_lactate_available * log_t
    if (transform %in% c("ph_aki_time", "ph_labs_time"))
      data$ph_aki_assessable <- data$exposure_aki_assessable * log_t
  } else if (transform == "matched_24h" || transform == "matched_48h") {
    stop("Prepare matched lag risk-set input locally before this batch run")
  } else {
    stop("Unknown transform: ", transform)
  }
  data
}

results <- vector("list", nrow(manifest))
for (i in seq_len(nrow(manifest))) {
  name <- manifest$model_name[[i]]
  message("Fitting ", name)
  data <- read_frame(manifest$input_path[[i]])
  data <- apply_score_delta(data, manifest$score_delta_path[[i]])
  data <- transform_frame(data, manifest$transform[[i]])
  if (!all(c("subject_id", "stay_id", "t_start", "t_stop", "event") %in% names(data)))
    stop(name, ": missing risk columns")
  if (anyNA(data[, c("subject_id", "stay_id", "t_start", "t_stop", "event")]))
    stop(name, ": missing risk value")
  if (!all(data$t_stop > data$t_start) || !all(data$event %in% 0:1))
    stop(name, ": invalid counting-process interval")
  if ("exposure_vulnerability_category" %in% names(data))
    data$exposure_vulnerability_category <- factor(data$exposure_vulnerability_category,
                                                    levels = c("low", "moderate", "high"))
  for (category in c("icu_group", "admission_group", "cvc_type_group"))
    if (category %in% names(data)) data[[category]] <- factor(data[[category]])
  formula <- as.formula(paste("Surv(t_start, t_stop, event) ~", manifest$rhs[[i]]))
  warnings_seen <- character()
  fit <- withCallingHandlers(
    coxph(formula, data = data, ties = "efron", robust = TRUE, cluster = subject_id,
          singular.ok = FALSE,
          control = coxph.control(eps = 1e-10, iter.max = 50, timefix = FALSE)),
    warning = function(w) {
      warnings_seen <<- c(warnings_seen, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  if (fit$n != nrow(data) || fit$nevent != sum(data$event)) stop(name, ": implicit complete-case deletion")
  beta <- coef(fit)
  covariance <- vcov(fit)
  se <- sqrt(diag(covariance))
  if (any(!is.finite(beta)) || any(!is.finite(covariance)) ||
      qr(covariance)$rank != length(beta)) stop(name, ": nonfinite or rank-deficient fit")
  z <- qnorm(0.975)
  coefficients <- data.frame(model = name, term = names(beta), coefficient = as.numeric(beta),
    robust_se = se, hazard_ratio = exp(beta), ci_lower = exp(beta - z * se),
    ci_upper = exp(beta + z * se), p_value = 2 * pnorm(-abs(beta / se)),
    intervals = nrow(data), events = sum(data$event), patients = length(unique(data$subject_id)),
    stays = length(unique(data$stay_id)), row.names = NULL)
  write.csv(coefficients, file.path(output_dir, paste0(name, "_coefficients.csv")), row.names = FALSE)
  write.csv(covariance, file.path(output_dir, paste0(name, "_covariance.csv")), row.names = TRUE)
  diagnostics <- data.frame(model = name, intervals = fit$n, events = fit$nevent,
    patients = length(unique(data$subject_id)), stays = length(unique(data$stay_id)),
    risk_person_days = if ("duration_days" %in% names(data)) sum(data$duration_days) else sum(data$t_stop - data$t_start),
    parameters = length(beta), iterations = paste(fit$iter, collapse = ","),
    partial_log_likelihood = fit$loglik[[2L]],
    robust_covariance_rank = qr(covariance)$rank,
    robust_covariance_min_eigenvalue = min(eigen(covariance, symmetric = TRUE, only.values = TRUE)$values),
    warning_text = paste(unique(warnings_seen), collapse = " | "),
    rhs = manifest$rhs[[i]], ties = "efron", cluster = "subject_id",
    r_version = R.version.string, survival_version = as.character(packageVersion("survival")),
    stringsAsFactors = FALSE)
  write.csv(diagnostics, file.path(output_dir, paste0(name, "_diagnostics.csv")), row.names = FALSE)
  results[[i]] <- diagnostics
  rm(data, fit, covariance, beta, se)
  gc(verbose = FALSE)
}
write.csv(do.call(rbind, results), file.path(output_dir, "batch_diagnostics.csv"), row.names = FALSE)
cat("Completed", nrow(manifest), "R survival Cox models; aggregates only\n")
