def get_feature_metrics_help_modal() -> str:
    modal_html = """
<div id="featureMetricsHelpModal" class="modal">
  <div class="modal-content">
    <span class="close-feature-metrics">&times;</span>
    <h2>Help Guide: Common Model Metrics</h2>
    <div class="metrics-guide" style="max-height:65vh;overflow-y:auto;font-size:1.04em;">
      <h3>1) General Metrics</h3>
      <h4>Classification</h4>
      <p><strong>Accuracy:</strong> The proportion of correct predictions among all predictions. It is calculated as (TP + TN) / (TP + TN + FP + FN). While intuitive, Accuracy can be misleading for imbalanced datasets where one class dominates. For example, in a dataset with 95% negative cases, a model predicting all negatives achieves 95% Accuracy but fails to identify positives.</p>
      <p><strong>AUC (Area Under the Curve):</strong> Specifically, the Area Under the Receiver Operating Characteristic Curve (ROC-AUC) measures a model’s ability to distinguish between classes. It ranges from 0 to 1, where 1 indicates perfect separation and 0.5 suggests random guessing. ROC-AUC is robust for binary and multiclass problems but may be less informative for highly imbalanced datasets.</p>
      <h4>Regression</h4>
      <p><strong>R2 (Coefficient of Determination):</strong> Measures the proportion of variance in the dependent variable explained by the independent variables. It ranges from 0 to 1, with 1 indicating perfect prediction and 0 indicating no explanatory power. Negative values are possible if the model performs worse than a mean-based baseline. R2 is widely used but sensitive to outliers.</p>
      <p><strong>RMSE (Root Mean Squared Error):</strong> The square root of the average squared differences between predicted and actual values. It penalizes larger errors more heavily and is expressed in the same units as the target variable, making it interpretable. Lower RMSE indicates better model performance.</p>
      <p><strong>MAE (Mean Absolute Error):</strong> The average of absolute differences between predicted and actual values. It is less sensitive to outliers than RMSE and provides a straightforward measure of average error magnitude. Lower MAE is better.</p>

      <h3>2) Precision, Recall & Specificity</h3>
      <h4>Classification</h4>
      <p><strong>Precision:</strong> The proportion of positive predictions that are correct, calculated as TP / (TP + FP). High Precision is crucial when false positives are costly, such as in spam email detection, where misclassifying legitimate emails as spam disrupts user experience.</p>
      <p><strong>Recall (Sensitivity):</strong> The proportion of actual positives correctly predicted, calculated as TP / (TP + FN). High Recall is vital when missing positives is risky, such as in disease diagnosis, where failing to identify a sick patient could have severe consequences.</p>
      <p><strong>Specificity:</strong> The true negative rate, calculated as TN / (TN + FP). It measures how well a model identifies negatives, making it valuable in medical testing to minimize false alarms (e.g., incorrectly diagnosing healthy patients as sick).</p>

      <h3>3) Macro, Micro, and Weighted Averages</h3>
      <h4>Classification</h4>
      <p><strong>Macro Precision / Recall / F1:</strong> Computes the metric for each class independently and averages them, treating all classes equally. This is ideal for balanced datasets or when all classes are equally important, such as in multiclass image classification with similar class frequencies.</p>
      <p><strong>Micro Precision / Recall / F1:</strong> Aggregates true positives (TP), false positives (FP), and false negatives (FN) across all classes before computing the metric. It provides a global perspective and is suitable for imbalanced datasets or multilabel problems, as it accounts for class frequency.</p>
      <p><strong>Weighted Precision / Recall / F1:</strong> Averages the metric across classes, weighted by the number of true instances per class. This balances the importance of classes based on their frequency, making it useful for imbalanced datasets where larger classes should have more influence but smaller classes are still considered.</p>

      <h3>4) Average Precision (PR-AUC Variants)</h3>
      <h4>Classification</h4>
      <p><strong>Average Precision:</strong> The Area Under the Precision-Recall Curve (PR-AUC) summarizes the trade-off between Precision and Recall. It is particularly useful for imbalanced datasets, where ROC-AUC may overestimate performance. Average Precision is computed by averaging Precision values at different Recall thresholds, providing a robust measure for ranking tasks or rare class detection.</p>

      <h3>5) ROC-AUC Variants</h3>
      <h4>Classification</h4>
      <p><strong>ROC-AUC:</strong> The Area Under the Receiver Operating Characteristic Curve plots the true positive rate (Recall) against the false positive rate (1 - Specificity) at various thresholds. It quantifies the model’s ability to separate classes, with higher values indicating better performance.</p>
      <p><strong>Macro ROC-AUC:</strong> Averages the ROC-AUC scores across all classes, treating each class equally. This is suitable for balanced multiclass problems where all classes are of equal importance.</p>
      <p><strong>Micro ROC-AUC:</strong> Computes a single ROC-AUC by aggregating predictions and true labels across all classes. It is effective for multiclass or multilabel problems with class imbalance, as it accounts for the overall prediction distribution.</p>

      <h3>6) Confusion Matrix Stats (Per Class)</h3>
      <h4>Classification</h4>
      <p><strong>True Positives (TP):</strong> The number of correct positive predictions for a given class.</p>
      <p><strong>True Negatives (TN):</strong> The number of correct negative predictions for a given class.</p>
      <p><strong>False Positives (FP):</strong> The number of incorrect positive predictions for a given class (false alarms).</p>
      <p><strong>False Negatives (FN):</strong> The number of incorrect negative predictions for a given class (missed detections). These stats are visualized in PyCaret’s confusion matrix plots, aiding class-wise performance analysis.</p>

      <h3>7) Other Useful Metrics</h3>
      <h4>Classification</h4>
      <p><strong>Cohen’s Kappa:</strong> Measures the agreement between predicted and actual labels, adjusted for chance. It ranges from -1 to 1, where 1 indicates perfect agreement, 0 indicates chance-level agreement, and negative values suggest worse-than-chance performance. Kappa is useful for multiclass problems with imbalanced labels.</p>
      <p><strong>Matthews Correlation Coefficient (MCC):</strong> A balanced measure that considers TP, TN, FP, and FN, calculated as (TP * TN - FP * FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN)). It ranges from -1 to 1, with 1 being perfect prediction. MCC is particularly effective for imbalanced datasets due to its symmetry across classes.</p>
      <h4>Regression</h4>
      <p><strong>MSE (Mean Squared Error):</strong> The average of squared differences between predicted and actual values. It amplifies larger errors, making it sensitive to outliers. Lower MSE indicates better performance.</p>
      <p><strong>MAPE (Mean Absolute Percentage Error):</strong> The average of absolute percentage differences between predicted and actual values, calculated as (1/n) * Σ(|actual - predicted| / |actual|) * 100. It is useful when relative errors are important but can be unstable if actual values are near zero.</p>
    </div>
  </div>
</div>
"""
    modal_css = """
<style>
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
.metrics-guide h3 { margin-top: 20px; }
.metrics-guide h4 { margin-top: 12px; color: #17623b; }
.metrics-guide p { margin: 5px 0 10px 0; }
.metrics-guide ul { margin: 10px 0 10px 24px; }
</style>
"""
    modal_js = """
<script>
document.addEventListener("DOMContentLoaded", function() {
  var modal = document.getElementById("featureMetricsHelpModal");
  var openBtn = document.getElementById("openFeatureMetricsHelp");
  var span = document.getElementsByClassName("close-feature-metrics")[0];
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
