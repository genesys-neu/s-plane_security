import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


lstm = np.array([[0.7, 0.3], [0.25, 0.75]])
tc232 = np.array([[0.9595, 0.0405], [0.0362, 0.9638]])
tc240 = np.array([[0.9908, 0.0092], [0.0320, .9680]])
tc332 = np.array([[0.9693, 0.0307], [0.0153, .9847]])
tc340 = np.array([[0.9749, 0.0251], [0.0167, .9833]])
t2tc232 = np.array([[0.9696, 0.0304], [0.0601, 0.9399]])
t2tc240 = np.array([[0.9347, 0.0653], [0.0277, .9723]])
t2tc332 = np.array([[0.8452, 0.1548], [0.0488, .9512]])
t2tc340 = np.array([[0.9752, 0.0248], [0.0315, .9685]])
overall_confusion_matrix_nobg = np.array([[0.9602, 0.0398], [0.0072, 0.9928]])
overall_confusion_matrix_bg = np.array([[0.4881, 0.5119], [0.0307, 0.9693]])
heuristic = np.array([[0.9947, 0.0053], [0.8220, 0.1780]])
toverall_confusion_matrix_bg = np.array([[0.8178, 0.1822], [0.0736, 0.9264]])
t7jul_lat_ext_9jul1 = np.array([[0.8089696, 0.1910304], [0.0522441, 0.9477559]])
prod = np.array([[0.9900, 0.0100], [0.0430, 0.9570]])
dt = np.array([[0.9525, 0.04750], [0.0301, 0.9699]])
cnn = np.array([[17289, 149], [752, 24779]])
print(cnn)
cnn = np.around(cnn/cnn.sum(axis=1), decimals=4)
print(cnn)
cnnprod = np.array([[0.9997, 0.0003], [0.6675, 0.3325]])
cnn_dt = np.array([[0.9998, 0.0002], [1, 0]])

# Plot the confusion matrix
plt.figure(figsize=(6, 3))
sns.heatmap(cnn_dt, annot=True, fmt='.2%', cmap='Blues', cbar=False, annot_kws={"size": 24})
plt.xlabel('Predicted Label', fontsize=18)
plt.ylabel('True Label', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Remove internal gridlines
plt.grid(False)

plt.gca().spines['top'].set_visible(True)  # show top line
plt.gca().spines['right'].set_visible(True)  # show right line
plt.gca().spines['left'].set_visible(True)  # show top line
plt.gca().spines['bottom'].set_visible(True)  # show right line
plt.tight_layout()
# plt.title(f'Confusion Matrix - Model {model_name}')
# Save the confusion matrix plot as a .png file
# plt.savefig(f"dt.png")
plt.savefig("cnn_dt.pdf", format='pdf')
plt.close()