# mock_fit.R -- group-lasso Poisson fit over every glm_input_*.mat in a folder
#
#   Rscript mock_fit.R <input folder> [criterion] [shard nshards]
#
# With shard/nshards (e.g. 2 3) only every nshards-th input is fitted, starting
# at the shard-th, so several R processes can split one folder between them.
# Each target is an independent fit, so the split does not change any result.
#
# Mock-test driver for new data. The model is IDENTICAL to the verified
# FittingGroupPoisson.R -- same grpreg call, penalty, family, 100-value lambda
# path and BIC selection -- but it loops over whatever inputs a folder holds
# instead of hand-typed patient/session numbers.
#
# One deliberate difference: X and y are NOT written back out. The shipped
# grpreg_*.mat files stored them, which is why they reach 1.4 GB each. The
# collector needs only Beta, Lambda, group and the selection.

suppressPackageStartupMessages({ library(grpreg); library(R.matlab) })

args      <- commandArgs(trailingOnly = TRUE)
indir     <- if (length(args) >= 1) args[1] else "."
criterion <- if (length(args) >= 2) args[2] else "BIC"
shard     <- if (length(args) >= 4) as.integer(args[3]) else 1L
nshards   <- if (length(args) >= 4) as.integer(args[4]) else 1L

files <- sort(list.files(indir, pattern = "^glm_input_.*\\.mat$", full.names = TRUE))
if (length(files) == 0) stop("no glm_input_*.mat files in ", indir)
files <- files[seq(shard, length(files), by = nshards)]

for (f in files) {
  t0  <- Sys.time()
  dat <- readMat(f)
  X <- as.matrix(dat$X); storage.mode(X) <- "double"
  y <- as.double(as.vector(dat$y)); group <- as.vector(dat$group)
  rm(dat)
  stopifnot(ncol(X) == length(group), nrow(X) == length(y))

  out <- sub("^glm_input_", "grpreg_", basename(f))

  # grpreg ABORTS (not warns) on very sparse targets -- e.g. "Unpenalized portion
  # of model is already saturated" at 5 events in 104,807 bins. Without this
  # guard one such target kills every target after it (problems.md P-26).
  fit <- tryCatch(
    grpreg(X, y, group = group, penalty = "grLasso", family = "poisson", nlambda = 100),
    error = function(e) e)
  if (inherits(fit, "error")) {
    writeMat(file.path(indir, out), fit_failed = 1, message = conditionMessage(fit),
             n_events = sum(y), n_bins = length(y), group = group)
    cat(sprintf("%-46s events=%5d  FAILED: %s\n", basename(f), as.integer(sum(y)),
                conditionMessage(fit)))
    next
  }
  sel <- select(fit, criterion = criterion)
  best_idx <- which.min(sel$IC)
  writeMat(file.path(indir, out),
           Lambda = as.vector(fit$lambda), Beta = fit$beta, group = group,
           BICs = sel$IC, best_idx = best_idx, criterion = criterion,
           n_events = sum(y), n_bins = length(y))

  edge <- if (best_idx == 1 || best_idx == length(fit$lambda)) "  <-- ENDPOINT" else ""
  cat(sprintf("%-46s events=%5d  %s idx=%3d/%d  %.1fs%s\n", basename(f), as.integer(sum(y)),
              criterion, best_idx, length(fit$lambda),
              as.numeric(difftime(Sys.time(), t0, units = "secs")), edge))
  rm(X, fit); invisible(gc())
}
