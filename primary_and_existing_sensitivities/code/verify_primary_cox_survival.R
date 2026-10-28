# Independent implementation check using R survival::coxph.  Reads only the
# local derived intervals and local stay-level coding delta; writes aggregates.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L || !(args[[1L]] %in% c("original", "final", "immune_excluded"))) {
  stop("Usage: Rscript verify_primary_cox_survival.R original|final|immune_excluded")
}
mode <- args[[1L]]
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
stopifnot(length(script_arg) == 1L)
script_path <- sub("^--file=", "", script_arg[[1L]])
root <- normalizePath(file.path(dirname(script_path), "..", ".."), winslash = "/", mustWork = TRUE)
base <- file.path(root, "ARIC_revision_20260920")
out <- file.path(root, "ARIC_revision_20261009", "statistics_audit")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
library(survival)
risk_path <- if (mode == "immune_excluded") {
  file.path(root, "ARIC_revision_20261009", "secure_derived_inputs",
            "immune_flag_excluded_day3_risk.csv.gz")
} else {
  file.path(base, "secure_derived_inputs", "revised_risk_day3.csv.gz")
}
risk <- read.csv(gzfile(risk_path),
                 stringsAsFactors = FALSE, check.names = FALSE)
stopifnot(nrow(risk) == 60272L, sum(risk$event) == 268L,
          length(unique(risk$subject_id)) == 12757L,
          all(risk$t_stop > risk$t_start))
if (mode == "final") {
  delta <- read.csv(gzfile(file.path(root, "ARIC_revision_20261009",
                                     "secure_derived_inputs", "final_score_deltas_stay.csv.gz")))
  index <- match(risk$stay_id, delta$stay_id)
  stopifnot(!anyNA(index), all(delta$delta_score == delta$delta_cancer + delta$delta_meta))
  score <- risk$exposure_vulnerability_burden_score + delta$delta_score[index]
  risk$exposure_vulnerability_category <- ifelse(score <= 1, "low",
                                                ifelse(score <= 3, "moderate", "high"))
}
risk$exposure_vulnerability_category <- factor(risk$exposure_vulnerability_category,
                                                levels = c("low", "moderate", "high"))
risk$icu_group <- factor(risk$icu_group)
risk$admission_group <- factor(risk$admission_group)
risk$cvc_type_group <- factor(risk$cvc_type_group)
formula <- Surv(t_start, t_stop, event) ~ exposure_vulnerability_category +
  exposure_indication_present + age_per_10y + female + icu_group +
  admission_group + multiple_line + cvc_type_group +
  exposure_antibiotic_exposure + exposure_lactate_available +
  exposure_aki_assessable
fit <- coxph(formula, data = risk, ties = "efron", robust = TRUE,
             cluster = subject_id,
             singular.ok = FALSE,
             control = coxph.control(eps = 1e-10, iter.max = 50, timefix = FALSE))
beta <- as.numeric(coef(fit))
terms <- names(coef(fit))
variance <- vcov(fit)
se <- sqrt(diag(variance))
table <- data.frame(term = terms, coefficient = beta, robust_se = se,
                    hazard_ratio = exp(beta),
                    ci_lower = exp(beta - qnorm(0.975) * se),
                    ci_upper = exp(beta + qnorm(0.975) * se),
                    p_value = 2 * pnorm(-abs(beta / se)),
                    stringsAsFactors = FALSE)
write.csv(table, file.path(out, paste0("r_survival_primary_", mode, "_full_precision.csv")),
          row.names = FALSE, na = "")
write.csv(variance, file.path(out, paste0("r_survival_primary_", mode, "_covariance.csv")),
          row.names = TRUE, na = "")
write.csv(fit$naive.var, file.path(out, paste0("r_survival_primary_", mode, "_naive_covariance.csv")),
          row.names = TRUE, na = "")
diagnostics <- data.frame(
  mode = mode, survival_version = as.character(packageVersion("survival")),
  r_version = R.version.string, intervals = nrow(risk),
  events = sum(risk$event), patients = length(unique(risk$subject_id)),
  stays = length(unique(risk$stay_id)), risk_days = sum(risk$duration_days),
  fitted_rows = fit$n, fitted_events = fit$nevent,
  loglik = fit$loglik[[2L]],
  iterations = fit$iter, n_parameters = length(beta),
  robust_variance_min_eigenvalue = min(eigen(variance, symmetric = TRUE,
                                            only.values = TRUE)$values),
  model_formula = paste(deparse(formula), collapse = " "),
  ties = "efron", cluster = "subject_id", timefix = FALSE,
  stringsAsFactors = FALSE)
write.csv(diagnostics, file.path(out, paste0("r_survival_primary_", mode, "_diagnostics.csv")),
          row.names = FALSE)
if (mode == "final") {
  # Schoenfeld-residual score diagnostic on the actual event-time axis. It is
  # complementary to the explicit patient-cluster robust interaction tests;
  # do not interpret a null score test as proof of proportional hazards.
  zph <- cox.zph(fit, transform = function(t) log1p(t), terms = TRUE, global = TRUE)
  zph_table <- data.frame(term = rownames(zph$table), zph$table, row.names = NULL,
                          check.names = FALSE)
  write.csv(zph_table, file.path(out, "r_survival_primary_final_cox_zph_log_event_time.csv"),
            row.names = FALSE)
}
cat(mode, "R survival::coxph complete;", nrow(risk), "intervals;", sum(risk$event),
    "events;", length(beta), "terms\n")
