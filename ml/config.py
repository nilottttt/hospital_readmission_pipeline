"""Settings for the ML pipeline, API and monitoring (all overridable via env vars)."""
import os

# --- MLflow ----------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "hospital_readmission_30d")
MODEL_NAME = os.getenv("MODEL_NAME", "readmission_risk_model")
CHAMPION_ALIAS = "champion"      # version served by the API
PREVIOUS_ALIAS = "previous"      # last champion, kept for rollback

# --- Training ----------------------------------------------------------------
RANDOM_STATE = 42
TRAIN_FRACTION = 0.70            # oldest 70 % of encounters
VALID_FRACTION = 0.15            # next 15 %  (model selection, calibration, threshold)
#                                  newest 15 % = test (final, untouched evaluation)
MIN_TRAINING_ROWS = 10_000
# A new model replaces the champion only if its test ROC-AUC is at least
# the champion's ROC-AUC on the SAME test set minus this tolerance.
PROMOTION_TOLERANCE = 0.002

# --- API ---------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")

# --- Monitoring & retraining criteria -----------------------------------------
MONITOR_WINDOW_DAYS = int(os.getenv("MONITOR_WINDOW_DAYS", "7"))
MONITOR_BATCH_SIZE = int(os.getenv("MONITOR_BATCH_SIZE", "2000"))
PSI_WARNING = 0.10               # moderate shift
PSI_ALERT = 0.25                 # significant shift  -> retrain
AUC_DROP_ALERT = 0.05            # ROC-AUC below champion test AUC by more -> retrain
ECE_ALERT = 0.10                 # expected calibration error above -> retrain
SUBGROUP_RECALL_GAP_ALERT = 0.20 # fairness review (not an automatic retrain)
ERROR_RATE_ALERT = 0.05          # API failures (operational alert)
P95_LATENCY_ALERT_MS = 500       # API latency (operational alert)
MIN_LABELED_FOR_PERFORMANCE = 200
SUBGROUP_MIN_N = 100             # subgroups smaller than this are not compared
SUBGROUP_MIN_POSITIVES = 20      # ... nor those with too few readmissions
RETRAIN_COOLDOWN_DAYS = int(os.getenv("RETRAIN_COOLDOWN_DAYS", "7"))       # never auto-retrain more often than this
