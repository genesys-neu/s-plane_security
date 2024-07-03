import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


# lstm = np.array([[0.7, 0.3], [0.25, 0.75]])
# tc232 = np.array([[0.9595, 0.0405], [0.0362, 0.9638]])
# tc240 = np.array([[0.9908, 0.0092], [0.0320, .9680]])
# tc332 = np.array([[0.9693, 0.0307], [0.0153, .9847]])
# tc340 = np.array([[0.9749, 0.0251], [0.0167, .9833]])
# overall_confusion_matrix_nobg = np.array([[0.9602, 0.0398], [0.0072, 0.9928]])
conf_matrix = np.array([[0.4881, 0.5119], [0.0307, 0.9693]])
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='.2%', cmap='Blues', cbar=False, annot_kws={"size": 24})
plt.xlabel('Predicted Label', fontsize=18)
plt.ylabel('True Label', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.title(f'Confusion Matrix - Model {model_name}')
# Save the confusion matrix plot as a .png file
plt.savefig(f"conf_matrix.png")
plt.close()