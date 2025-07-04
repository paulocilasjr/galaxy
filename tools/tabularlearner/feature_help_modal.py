def get_feature_metrics_help_modal() -> str:
    # 1) Button HTML (styled via the shared CSS below)
    button_html = """
<button class="help-modal-btn" id="openFeatureMetricsHelp">
  Help: Metrics Guide
</button>
"""

    # 2) Shared button + modal CSS
    modal_css = """
<style>
/* Shared Help-Button Styling (copied from image_learner_cli) */
.help-modal-btn {
  background-color: #17623b;
  color: #fff;
  border: none;
  border-radius: 24px;
  padding: 10px 28px;
  font-size: 1.1rem;
  font-weight: bold;
  letter-spacing: 0.03em;
  cursor: pointer;
  transition: background 0.2s, box-shadow 0.2s;
  box-shadow: 0 2px 8px rgba(23,98,59,0.07);
  margin-left: auto; /* if used within a flex container */
}
.help-modal-btn:hover,
.help-modal-btn:focus {
  background-color: #21895e;
  box-shadow: 0 4px 16px rgba(23,98,59,0.14);
}

/* Modal Background & Content */
#featureMetricsHelpModal.modal {
  display: none;
  position: fixed;
  z-index: 9999;
  left: 0; top: 0;
  width: 100%; height: 100%;
  overflow: auto;
  background-color: rgba(0,0,0,0.45);
}
#featureMetricsHelpModal .modal-content {
  background-color: #fefefe;
  margin: 5% auto;
  padding: 24px 28px 20px 28px;
  border: 1.5px solid #17623b;
  width: 90%;
  max-width: 800px;
  border-radius: 18px;
  box-shadow: 0 8px 32px rgba(23,98,59,0.20);
}
#featureMetricsHelpModal .close-feature-metrics {
  color: #17623b;
  float: right;
  font-size: 28px;
  font-weight: bold;
  cursor: pointer;
  transition: color 0.2s;
}
#featureMetricsHelpModal .close-feature-metrics:hover {
  color: #21895e;
}

/* Inner scrollable content */
.metrics-guide {
  max-height: 65vh;
  overflow-y: auto;
  font-size: 1.04em;
}
.metrics-guide h3 { margin-top: 20px; }
.metrics-guide h4 { margin-top: 12px; color: #17623b; }
.metrics-guide p { margin: 5px 0 10px 0; }
.metrics-guide ul { margin: 10px 0 10px 24px; }
</style>
"""

    # 3) Modal HTML
    modal_html = """
<div id="featureMetricsHelpModal" class="modal">
  <div class="modal-content">
    <span class="close-feature-metrics">&times;</span>
    <h2>Help Guide: Common Model Metrics</h2>
    <div class="metrics-guide">
      <h3>1) General Metrics</h3>
      <h4>Classification</h4>
      <p><strong>Accuracy:</strong> The proportion of correct predictions among all predictions. It is calculated as (TP + TN) / (TP + TN + FP + FN). While intuitive, Accuracy can be misleading for imbalanced datasets where one class dominates. For example, in a dataset with 95% negative cases, a model predicting all negatives achieves 95% Accuracy but fails to identify positives.</p>
      <p><strong>AUC (Area Under the Curve):</strong> Specifically, the Area Under the Receiver Operating Characteristic Curve (ROC-AUC) measures a model’s ability to distinguish between classes. It ranges from 0 to 1, where 1 indicates perfect separation and 0.5 suggests random guessing. ROC-AUC is robust for binary and multiclass problems but may be less informative for highly imbalanced datasets.</p>
      <h4>Regression</h4>
      <p><strong>R2 (Coefficient of Determination):</strong> Measures the proportion of variance in the dependent variable explained by the independent variables. It ranges from 0 to 1, with 1 indicating perfect prediction and 0 indicating no explanatory power. Negative values are possible if the model performs worse than a mean-based baseline. R2 is widely used but sensitive to outliers.</p>
      <p><strong>RMSE (Root Mean Squared Error):</strong> The square root of the average squared differences between predicted and actual values. It penalizes larger errors more heavily and is expressed in the same units as the target variable, making it interpretable. Lower RMSE indicates better model performance.</p>
      <p><strong>MAE (Mean Absolute Error):</strong> The average of absolute differences between predicted and actual values. It is less sensitive to outliers than RMSE and provides a straightforward measure of average error magnitude. Lower MAE is better.</p>

      <!-- …rest of your metrics guide… -->

      <h4>Regression</h4>
      <p><strong>MSE (Mean Squared Error):</strong> The average of squared differences between predicted and actual values. It amplifies larger errors, making it sensitive to outliers. Lower MSE indicates better performance.</p>
      <p><strong>MAPE (Mean Absolute Percentage Error):</strong> The average of absolute percentage differences between predicted and actual values, calculated as (1/n) * Σ(|actual - predicted| / |actual|) * 100. It is useful when relative errors are important but can be unstable if actual values are near zero.</p>
    </div>
  </div>
</div>
"""

    # 4) Modal JS
    modal_js = """
<script>
document.addEventListener("DOMContentLoaded", function() {
  var modal = document.getElementById("featureMetricsHelpModal");
  var openBtn = document.getElementById("openFeatureMetricsHelp");
  var closeSpan = document.getElementsByClassName("close-feature-metrics")[0];
  if (openBtn && modal) {
    openBtn.onclick = function() { modal.style.display = "block"; };
  }
  if (closeSpan && modal) {
    closeSpan.onclick = function() { modal.style.display = "none"; };
  }
  window.onclick = function(event) {
    if (event.target == modal) {
      modal.style.display = "none";
    }
  };
});
</script>
"""

    # 5) Return combined
    return button_html + modal_css + modal_html + modal_js

