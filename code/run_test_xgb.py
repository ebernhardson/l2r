from utils import time_utils
from run_data import PythonRunner

if __name__ == "__main__":
    suffix = time_utils._timestamp()
    threshold = 0.10
    feat_conf = "feature_conf_linear_%s" % (suffix)
    combined_feat = "basic_linear_%s" % (suffix)

    PythonRunner()([
        # Generate feature config
        ["get_feature_conf_linear.py", "-d", "10", "-o", feat_conf],
        # Merge features
        ["feature_combiner.py", "-c", feat_conf, "-n", combined_feat, "-t", str(threshold)],
        # Train GradientBoostingRegressor
        ["task.py", "-f", combined_feat, "-l", "reg_xgb_rank", "-e", "100"],
    ])

