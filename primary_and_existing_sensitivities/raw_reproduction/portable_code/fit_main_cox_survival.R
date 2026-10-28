# Fit the single corrected main counting-process Cox model with patient clustering.
# Input contains restricted intervals and remains local. Outputs are aggregate only.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript fit_main_cox_survival.R OUTPUT_ROOT")
root <- normalizePath(args[[1L]], winslash = "/", mustWork = TRUE)
secure <- file.path(root, "secure_derived_inputs")
out <- file.path(root, "aggregate")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
library(survival)
risk <- read.csv(gzfile(file.path(secure, "risk_final_day3.csv.gz")),
                 stringsAsFactors = FALSE, check.names = FALSE)
stopifnot(nrow(risk) > 0L, sum(risk$event) > 0L,
          all(risk$t_stop > risk$t_start),
          all(is.finite(risk$t_start)), all(is.finite(risk$t_stop)))
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
             cluster = subject_id, singular.ok = FALSE,
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
                    intervals = nrow(risk), events = sum(risk$event),
                    patients = length(unique(risk$subject_id)),
                    stringsAsFactors = FALSE)
write.csv(table, file.path(out, "raw_reproduced_main_cox_full_precision.csv"),
          row.names = FALSE, na = "")
write.csv(variance, file.path(out, "raw_reproduced_main_cox_covariance.csv"),
          row.names = TRUE, na = "")
write.csv(fit$naive.var, file.path(out, "raw_reproduced_main_cox_naive_covariance.csv"),
          row.names = TRUE, na = "")
diagnostics <- data.frame(
  survival_version = as.character(packageVersion("survival")),
  r_version = R.version.string, intervals = nrow(risk),
  events = sum(risk$event), patients = length(unique(risk$subject_id)),
  stays = length(unique(risk$stay_id)), risk_days = sum(risk$duration_days),
  fitted_rows = fit$n, fitted_events = fit$nevent,
  loglik = fit$loglik[[2L]], iterations = fit$iter,
  n_parameters = length(beta),
  robust_variance_min_eigenvalue = min(eigen(variance, symmetric = TRUE,
                                            only.values = TRUE)$values),
  model_formula = paste(deparse(formula), collapse = " "),
  ties = "efron", cluster = "subject_id", timefix = FALSE,
  stringsAsFactors = FALSE)
write.csv(diagnostics, file.path(out, "raw_reproduced_main_cox_diagnostics.csv"),
          row.names = FALSE)
zph <- cox.zph(fit, transform = function(t) log1p(t), terms = TRUE, global = TRUE)
zph_table <- data.frame(term = rownames(zph$table), zph$table, row.names = NULL,
                        check.names = FALSE)
write.csv(zph_table, file.path(out, "raw_reproduced_main_cox_zph_log_event_time.csv"),
          row.names = FALSE)
cat("R survival::coxph complete;", nrow(risk), "intervals;", sum(risk$event),
    "events;", length(beta), "terms\n")
