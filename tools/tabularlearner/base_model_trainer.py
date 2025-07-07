import base64
import logging
import os
import tempfile
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from feature_help_modal import get_feature_metrics_help_modal
from feature_importance import FeatureImportanceAnalyzer
from sklearn.metrics import average_precision_score
from utils import get_html_template, build_tabbed_html, get_html_closing, encode_image_to_base64

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


class BaseModelTrainer:
    def __init__(
        self,
        input_file,
        target_col,
        output_dir,
        task_type,
        random_seed,
        test_file=None,
        **kwargs,
    ):
        self.exp = None
        self.input_file = input_file
        self.target_col = target_col
        self.output_dir = output_dir
        self.task_type = task_type
        self.random_seed = random_seed
        self.data = None
        self.target = None
        self.best_model = None
        self.results = None
        self.features_name = None
        self.plots = {}
        self.plots_explainer_html = None
        self.trees = []
        self.user_kwargs = kwargs.copy()
        for key, value in self.user_kwargs.items():
            setattr(self, key, value)
        self.setup_params = {}
        self.test_file = test_file
        self.test_data = None

        if not self.output_dir:
            raise ValueError("output_dir must be specified and not None")

        LOG.info(f"Model kwargs: {self.__dict__}")

    def load_data(self):
        LOG.info(f"Loading data from {self.input_file}")
        self.data = pd.read_csv(self.input_file, sep=None, engine="python")
        self.data.columns = self.data.columns.str.replace(".", "_")
        if "prediction_label" in self.data.columns:
            self.data = self.data.drop(columns=["prediction_label"])

        numeric_cols = self.data.select_dtypes(include=["number"]).columns
        non_numeric_cols = self.data.select_dtypes(exclude=["number"]).columns
        self.data[numeric_cols] = self.data[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        if len(non_numeric_cols) > 0:
            LOG.info(f"Non-numeric columns found: {non_numeric_cols.tolist()}")

        names = self.data.columns.to_list()
        target_index = int(self.target_col) - 1
        self.target = names[target_index]
        self.features_name = [n for i, n in enumerate(names) if i != target_index]

        if getattr(self, "missing_value_strategy", None):
            strat = self.missing_value_strategy
            if strat == "mean":
                self.data = self.data.fillna(self.data.mean(numeric_only=True))
            elif strat == "median":
                self.data = self.data.fillna(self.data.median(numeric_only=True))
            elif strat == "drop":
                self.data = self.data.dropna()
        else:
            self.data = self.data.fillna(self.data.median(numeric_only=True))

        if self.test_file:
            LOG.info(f"Loading test data from {self.test_file}")
            df_test = pd.read_csv(self.test_file, sep=None, engine="python")
            df_test.columns = df_test.columns.str.replace(".", "_")
            self.test_data = df_test

    def setup_pycaret(self):
        LOG.info("Initializing PyCaret")
        self.setup_params = {
            "target": self.target,
            "session_id": self.random_seed,
            "html": True,
            "log_experiment": False,
            "system_log": False,
            "index": False,
        }
        if self.test_data is not None:
            self.setup_params["test_data"] = self.test_data
        for attr in [
            "train_size",
            "normalize",
            "feature_selection",
            "cross_validation",
            "remove_outliers",
            "remove_multicollinearity",
            "polynomial_features",
            "fix_imbalance",
        ]:
            val = getattr(self, attr, None)
            if val is not None:
                self.setup_params[attr] = val
        if getattr(self, "cross_validation", None) and getattr(
            self, "cross_validation_folds", None
        ):
            self.setup_params["fold"] = self.cross_validation_folds
        LOG.info(self.setup_params)

        if self.task_type == "classification":
            from pycaret.classification import ClassificationExperiment

            self.exp = ClassificationExperiment()
        elif self.task_type == "regression":
            from pycaret.regression import RegressionExperiment

            self.exp = RegressionExperiment()
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")

        self.exp.setup(self.data, **self.setup_params)
        self.setup_params.update(self.user_kwargs)

    def train_model(self):
        LOG.info("Training and selecting the best model")
        if self.task_type == "classification":
            self.exp.add_metric(
                id="PR-AUC-Weighted",
                name="PR-AUC-Weighted",
                target="pred_proba",
                score_func=average_precision_score,
                average="weighted",
            )
        self.best_model = (
            self.exp.compare_models(include=self.models)
            if getattr(self, "models", None)
            else self.exp.compare_models()
        )
        self.results = self.exp.pull()
        if self.task_type == "classification":
            self.results.rename(columns={"AUC": "ROC-AUC"}, inplace=True)
        _ = self.exp.predict_model(self.best_model)
        self.test_result_df = self.exp.pull()
        if self.task_type == "classification":
            self.test_result_df.rename(columns={"AUC": "ROC-AUC"}, inplace=True)

    def save_model(self):
        hdf5_path = Path(self.output_dir) / "pycaret_model.h5"
        with h5py.File(hdf5_path, "w") as f:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                joblib.dump(self.best_model, tmp.name)
                tmp.seek(0)
                model_bytes = tmp.read()
            f.create_dataset("model", data=np.void(model_bytes))

    def generate_plots(self):
        raise NotImplementedError("Subclasses should implement this method")

    def encode_image_to_base64(self, img_path: str) -> str:
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def save_html_report(self):
        LOG.info("Saving HTML report")

        # 1) Determine best model name
        try:
            best_model_name = str(self.results.iloc[0]["Model"])
        except Exception:
            best_model_name = type(self.best_model).__name__
        LOG.info(f"Best model determined as: {best_model_name}")

        # 2) Compute number of training samples
        try:
            n_train = self.exp.X_train.shape[0]
        except Exception:
            n_train = getattr(self.exp, "X_train_transformed", pd.DataFrame()).shape[0]

        # 3) Build setup‐params table
        all_params = self.setup_params  # includes PyCaret defaults + user_kwargs
        display_keys = [
            "Target",
            "Session ID",
            "Train Size",
            "Normalize",
            "Feature Selection",
            "Cross Validation",
            "Cross Validation Folds",
            "Remove Outliers",
            "Remove Multicollinearity",
            "Polynomial Features",
            "Fix Imbalance",
            "Models",
        ]
        setup_rows = []
        total_rows = self.data.shape[0]

        for key in display_keys:
            param_key = key.lower().replace(" ", "_")
            val = all_params.get(param_key)

            if key == "Train Size":
                # if user passed a fraction, use that; otherwise compute it
                if val is not None:
                    frac = float(val)
                else:
                    frac = n_train / total_rows if total_rows else 0
                display_val = f"{frac:.2f} ({n_train} rows)"
            elif key in {
                "Normalize",
                "Feature Selection",
                "Cross Validation",
                "Remove Outliers",
                "Remove Multicollinearity",
                "Polynomial Features",
                "Fix Imbalance",
            }:
                display_val = bool(val)
            elif key == "Cross Validation Folds":
                display_val = val if val is not None else "None"
            elif key == "Models":
                if isinstance(val, (list, tuple)):
                    display_val = ", ".join(map(str, val))
                else:
                    display_val = "None"
            else:
                display_val = val if val is not None else "None"

            setup_rows.append([key, display_val])

        # Optionally append the CV fold metric
        if hasattr(self.exp, "_fold_metric"):
            setup_rows.append(["best_model_metric", self.exp._fold_metric])

        df_setup = pd.DataFrame(setup_rows, columns=["Parameter", "Value"])
        df_setup.to_csv(Path(self.output_dir) / "setup_params.csv", index=False)

        # 4) Persist comparison & test results
        self.results.to_csv(Path(self.output_dir) / "comparison_results.csv", index=False)
        self.test_result_df.to_csv(Path(self.output_dir) / "test_results.csv", index=False)

        # 5) Persist best‐model parameters
        df_model = pd.DataFrame(
            self.best_model.get_params().items(),
            columns=["Parameter", "Value"]
        )
        df_model.to_csv(Path(self.output_dir) / "best_model.csv", index=False)

        # 6) Build HTML tabs
        header = f"<h2>Best Model: {best_model_name}</h2>"

        summary_html = (
            header
            + "<h3>Validation Result Metrics</h3>"
            + '<div class="table-wrapper">'
            + self.results.to_html(index=False, classes="table sortable")
            + '</div>'
            + "<h3>Setup Parameters</h3>"
            + '<div class="table-wrapper">'
            + df_setup.to_html(index=False, classes="table sortable")
            + '</div>'
        )

        test_html = (
            header
            + "<h3>Test Metrics</h3>"
            + self.test_result_df.to_html(index=False, classes="table sortable")
            + "<h3>Visualizations</h3>"
        )
        for idx, (name, path) in enumerate(self.plots.items(), start=1):
            b64 = encode_image_to_base64(path)
            test_html += (
                '<div class="plot">'
                f"<h4>{name.replace('_', ' ').title()}</h4>"
                f'<img src="data:image/png;base64,{b64}" '
                'style="max-width:90%;max-height:600px;border:1px solid #ddd;"/>'
                "</div>"
            )
            if idx < len(self.plots):
                test_html += "<hr>"

        feature_html = (
            header
            + "<h3>Feature Importance</h3>"
            + FeatureImportanceAnalyzer(
                data=self.data,
                target_col=self.target_col,
                task_type=self.task_type,
                output_dir=self.output_dir,
                exp=self.exp,
                best_model=self.best_model,
            ).run()
        )

        explainer_html = None
        if self.plots_explainer_html:
            explainer_html = header + "<h3>Explainer Plots</h3>" + self.plots_explainer_html
            for i, tree_b64 in enumerate(self.trees, start=1):
                explainer_html += (
                    '<div class="plot">'
                    f"<h4>Tree {i}</h4>"
                    f'<img src="data:image/png;base64,{tree_b64}" '
                    'style="max-width:90%;max-height:600px;border:1px solid #ddd;"/>'
                    "</div>"
                )

        # 7) Assemble and write the HTML report
        html = get_html_template()
        html += "<h1>Tabular Learner Model Report</h1>"
        html += build_tabbed_html(summary_html, test_html, feature_html, explainer_html)
        html += get_feature_metrics_help_modal()
        html += get_html_closing()

        report_path = Path(self.output_dir) / "comparison_result.html"
        report_path.write_text(html, encoding="utf-8")
        LOG.info(f"HTML report generated at: {report_path}")

    def save_dashboard(self):
        raise NotImplementedError("Subclasses should implement this method")

    def generate_plots_explainer(self):
        raise NotImplementedError("Subclasses should implement this method")

    def generate_tree_plots(self):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from xgboost import XGBClassifier, XGBRegressor
        from explainerdashboard.explainers import RandomForestExplainer

        LOG.info("Generating tree plots")
        X_test = self.exp.X_test_transformed.copy()
        y_test = self.exp.y_test_transformed

        if isinstance(self.best_model, (RandomForestClassifier, RandomForestRegressor)):
            n_trees = self.best_model.n_estimators
        elif isinstance(self.best_model, (XGBClassifier, XGBRegressor)):
            n_trees = len(self.best_model.get_booster().get_dump())
        else:
            LOG.warning("Tree plots not supported for this model type.")
            return

        explainer = RandomForestExplainer(self.best_model, X_test, y_test)
        for i in range(n_trees):
            fig = explainer.decisiontree_encoded(tree_idx=i, index=0)
            self.trees.append(fig)

    def run(self):
        self.load_data()
        self.setup_pycaret()
        self.train_model()
        self.save_model()
        self.generate_plots()
        self.generate_plots_explainer()
        self.generate_tree_plots()
        self.save_html_report()
        # self.save_dashboard()
