import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or 'QtAgg' depending on your system
 # ✅ Use non-GUI backend to prevent Qt errors

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate_model(model, X_test, y_test, label_names):
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {acc:.2f}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=label_names))
import matplotlib.pyplot as plt

def plot_history(history):
    # Accuracy plot
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_plot.png")  # 🔸 Save instead of show
    plt.close()

    # Loss plot
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")  # 🔸 Save instead of show
    plt.close()
