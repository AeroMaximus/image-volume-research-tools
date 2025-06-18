import numpy as np
from scipy.stats import shapiro, ttest_rel

# Example F1 scores for two deep learning algorithms

# Section 3.3.3
f1_model1 = np.array([0.5141, 0.7434, 0.7951, 0.6070, 0.7638, 0.7046]) # U-Net
f1_model2 = np.array([0.6181, 0.7742, 0.8783, 0.6927, 0.7523, 0.7618]) # FC-DenseNet

# # Section 3.4.2 without porosity
# f1_model1 = np.array([0.967869828646799, 0.969090687794502, 0.950787402]) # U-Net
# f1_model2 = np.array([0.924711201, 0.933945529, 0.867477504]) # FC-DenseNet

# # Section 3.4.2 with porosity
# f1_model1 = np.array([0.967869828646799, 0.969090687794502, 0.950787402, 0.960860342]) # U-Net
# f1_model2 = np.array([0.924711201, 0.933945529, 0.867477504, 0.80475027]) # FC-DenseNet

alpha = 0.05 # Significance

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
if p_value > alpha:
    print(f"The differences appear to be normally distributed "
          f"(p > {alpha:0.2f} so the null hypothesis of normality isn't rejected). You can use a paired t-test.")

    # Perform paired t-test
    t_stat, p_value = ttest_rel(f1_model1, f1_model2)

    # Print results
    print(f"Paired t-test Statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Interpretation
    if p_value > alpha:
        print(f"No significant difference between the models (p > {alpha:0.2f} so the null hypothesis isn't rejected).")
    else:
        print(f"Significant difference between the models (p <= {alpha:0.2f}).")

else:
    print(f"The differences do not appear to be normally distributed (p <= {alpha:0.2f}).")
