import base64
import logging
import os

import shap
import matplotlib.pyplot as plt
import pandas as pd
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    def __init__(
        self,
        task_type,
        output_dir,
        data_path=None,
        data=None,
        target_col=None,
        exp=None,
        best_model=None,
    ):
        self.task_type       = task_type
        self.output_dir      = output_dir
        self.exp             = exp
        self.best_model      = best_model
        self.tree_model_name = None
        self.shap_model_name = None

        if exp is not None:
            # use provided PyCaret experiment (already set up)
            self.data   = exp.dataset.copy()
            self.target = exp.target_param
            LOG.info("Using provided experiment object")
        else:
            if data is not None:
                self.data = data
                LOG.info("Data loaded from memory")
            else:
                self.target_col = target_col
                self.data = pd.read_csv(data_path, sep=None, engine="python")
                self.data.columns = self.data.columns.str.replace(".", "_")
                self.data = self.data.fillna(self.data.median(numeric_only=True))
            self.target = self.data.columns[int(target_col) - 1]
            self.exp    = (
                ClassificationExperiment()
                if task_type == "classification"
                else RegressionExperiment()
            )

        self.plots = {}

    def setup_pycaret(self):
        # Only run setup if exp not already set up
        if getattr(self.exp, "is_setup", False):
            LOG.info("Experiment already set up. Skipping PyCaret setup.")
            return
        LOG.info("Initializing PyCaret")
        setup_params = {
            "target":         self.target,
            "session_id":     123,
            "html":           True,
            "log_experiment": False,
            "system_log":     False,
        }
        self.exp.setup(self.data, **setup_params)

    def save_tree_importance(self):
        # prefer user’s best_model, else retrain random forest
        model = self.best_model or self.exp.create_model("rf")
        model_type = model.__class__.__name__
        self.tree_model_name = model_type

        processed_features = self.exp.get_config("X_transformed").columns

        # try tree‐based or linear‐coef models
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = abs(model.coef_).flatten()
        else:
            LOG.warning(f"{model_type} has no feature_importances_ or coef_; skipping.")
            self.tree_model_name = None
            return

        # guard against mismatch
        if len(importances) != len(processed_features):
            LOG.warning(
                f"Importances length {len(importances)} != features {len(processed_features)}; skipping."
            )
            self.tree_model_name = None
            return

        df_imp = (
            pd.DataFrame({
                "Feature":    processed_features,
                "Importance": importances,
            })
            .sort_values("Importance", ascending=False)
        )

        plt.figure(figsize=(10, 6))
        plt.barh(df_imp["Feature"], df_imp["Importance"])
        plt.xlabel("Importance")
        plt.title(f"Feature Importance ({model_type})")
        path = os.path.join(self.output_dir, "tree_importance.png")
        plt.savefig(path)
        plt.close()
        self.plots["tree_importance"] = path

    def save_shap_values(self):
        # pick best_model if present, else default LightGBM
        model = self.best_model or self.exp.create_model("lightgbm")
        X_trans = self.exp.get_config("X_transformed")
        model_class = model.__class__.__name__
        self.shap_model_name = model_class

        tree_like = (
            "LGBM", "XGB", "CatBoost", "RandomForest",
            "DecisionTree", "ExtraTrees", "HistGradientBoosting"
        )
        if any(tc in model_class for tc in tree_like):
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)
            plot_data   = X_trans
            plot_title  = f"SHAP Summary for {model_class} (TreeExplainer)"
        else:
            sampled     = X_trans.sample(100, random_state=42)
            explainer   = shap.KernelExplainer(model.predict, sampled)
            shap_values = explainer.shap_values(sampled)
            plot_data   = sampled
            plot_title  = f"SHAP Summary for {model_class} (KernelExplainer)"

        shap.summary_plot(shap_values, plot_data, show=False)
        plt.title(plot_title)
        path = os.path.join(self.output_dir, "shap_summary.png")
        plt.savefig(path)
        plt.close()
        self.plots["shap_summary"] = path

    def generate_feature_importance(self):
        self.save_tree_importance()
        self.save_shap_values()

    def encode_image_to_base64(self, img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate_html_report(self):
        LOG.info("Generating HTML report")

        plots_html = ""
        for name, path in self.plots.items():
            if name == "tree_importance" and not self.tree_model_name:
                continue
            encoded = self.encode_image_to_base64(path)

            if name == "tree_importance":
                title    = f"Feature importance analysis from a trained {self.tree_model_name}"
                subtitle = "Uses Gini impurity for classification; variance reduction for regression"
            elif name == "shap_summary":
                title    = f"SHAP Summary from a trained {self.shap_model_name}"
                subtitle = ""
            else:
                title, subtitle = name, ""

            plots_html += f"""
            <div class="plot" id="{name}">
                <h2>{title}</h2>
                {f"<h3>{subtitle}</h3>" if subtitle else ""}
                <img src="data:image/png;base64,{encoded}" alt="{name}">
            </div>
            """

        return f"""
        <h1>PyCaret Feature Importance Report</h1>
        {plots_html}
        """

    def run(self):
        if self.exp is None or not getattr(self.exp, "is_setup", False):
            self.setup_pycaret()
        self.generate_feature_importance()
        return self.generate_html_report()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature Importance Analysis")
    parser.add_argument("--data_path",   type=str, help="Path to the dataset")
    parser.add_argument("--target_col",  type=int, help="Index of target column (1-based)")
    parser.add_argument(
        "--task_type",
        choices=["classification", "regression"],
        help="Task type"
    )
    parser.add_argument("--output_dir",  type=str, help="Directory to save outputs")
    args = parser.parse_args()

    analyzer = FeatureImportanceAnalyzer(
        task_type   = args.task_type,
        output_dir  = args.output_dir,
        data_path   = args.data_path,
        target_col  = args.target_col,
        exp         = None,
        best_model  = None
    )
    html = analyzer.run()
    print(html)
