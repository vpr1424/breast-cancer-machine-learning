from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import randomForest,SVM,XGBoost,logReg , sNN, dNN
import numpy as np
import matplotlib.pyplot as plt
# load data
data = load_breast_cancer()
X = data.data
y = data.target

# test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models=["randomForest", "SVM", "XGBoost", "logReg", "sNN", "dNN"]
functions= {"randomForest":randomForest, "SVM":SVM, "XGBoost":XGBoost, "logReg":logReg, "sNN":sNN, "dNN":dNN}

fig, axes = plt.subplots(2, 3, figsize=(16, 8))

accuracies = []

for i,modelName in enumerate(models):
    modelFunction = functions[modelName]
    trainedModel = modelFunction(X_train,y_train)

    y_pred = np.rint(trainedModel.predict(X_test))

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plotting confusion matrix
    ax = axes[i % 2, i // 2]
    ax.set_title(f"Confusion Matrix for {modelName}")
    plt.sca(ax)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues, )
    plt.colorbar()
    tick_marks = np.arange(len(conf_matrix))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

plt.tight_layout()

fig2, ax = plt.subplots(1,1)
ax.bar(np.arange(len(models)),accuracies)
ax.set_xticks(np.arange(len(models)))
ax.set_xticklabels(models)
# ax.set_yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
ax.set_ylim([0.8,1.01])
ax.grid('True')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy across models')
plt.show()
