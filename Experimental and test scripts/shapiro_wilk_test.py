import numpy as np
from scipy.stats import shapiro, ttest_rel

# Example F1 scores for two deep learning algorithms
f1_model1 = np.array([0.5141, 0.7434, 0.7951, 0.6070, 0.7638, 0.7046]) # U-Net
f1_model2 = np.array([0.6181, 0.7742, 0.8783, 0.6927, 0.7523, 0.7618]) # FC-DenseNet

print(f"Average U-Net F1 score: {np.mean(f1_model1):.4f}")
print(f"Average FC-DenseNet F1 score: {np.mean(f1_model2):.4f}")

# Compute differences
differences = f1_model1 - f1_model2

# Perform Shapiro-Wilk test
stat, p_value = shapiro(differences)

# Print results
print(f"Shapiro-Wilk Test Statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
if p_value > 0.05:
    print("The differences appear to be normally distributed (p > 0.05). You can use a paired t-test.")

    # Perform paired t-test
    t_stat, p_value = ttest_rel(f1_model1, f1_model2)

    # Print results
    print(f"Paired t-test Statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Interpretation
    alpha = 0.05  # Significance level
    if p_value > alpha:
        print("No significant difference between the models (p > 0.05).")
    else:
        print("Significant difference between the models (p <= 0.05).")

else:
    print("The differences do not appear to be normally distributed (p <= 0.05).")
