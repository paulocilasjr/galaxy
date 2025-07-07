def get_feature_metrics_help_modal() -> str:
    modal_html = """
<div id="metricsHelpModal" class="modal">
  <div class="modal-content">
    <span class="close">&times;</span>
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
    modal_css = """
<style>
.modal {
  display: none;
  position: fixed;
  z-index: 1;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  overflow: auto;
  background-color: rgba(0,0,0,0.4);
}
.modal-content {
  background-color: #fefefe;
  margin: 15% auto;
  padding: 20px;
  border: 1px solid #888;
  width: 80%;
  max-width: 800px;
}
.close {
  color: #aaa;
  float: right;
  font-size: 28px;
  font-weight: bold;
}
.close:hover,
.close:focus {
  color: black;
  text-decoration: none;
  cursor: pointer;
}
.metrics-guide h3 {
  margin-top: 20px;
}
.metrics-guide p {
  margin: 5px 0;
}
.metrics-guide ul {
  margin: 10px 0;
  padding-left: 20px;
}
</style>
"""
    modal_js = """
<script>
document.addEventListener("DOMContentLoaded", function() {
  var modal = document.getElementById("metricsHelpModal");
  var openBtn = document.getElementById("openMetricsHelp");
  var span = document.getElementsByClassName("close")[0];
  if (openBtn && modal) {
    openBtn.onclick = function() {
      modal.style.display = "block";
    };
  }
  if (span && modal) {
    span.onclick = function() {
      modal.style.display = "none";
    };
  }
  window.onclick = function(event) {
    if (event.target == modal) {
      modal.style.display = "none";
    }
  }
});
</script>
"""
    return modal_css + modal_html + modal_js
