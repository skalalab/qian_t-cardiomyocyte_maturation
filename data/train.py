import numpy as np 
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

def prepare_data(df, splits):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # y = label_binarize(y, classes=np.unique(y))
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=1-splits, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix_roc(y_test, y_score, y_pred):
    classes = np.unique(y_test)
    num_classes = len(classes)
    cm = confusion_matrix(y_test, y_pred)
    y_test = label_binarize(y_test, classes=classes)

    # calculate roc curve
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_score[:,1])  # Probabilities for the positive class
        roc_auc = auc(fpr, tpr)
    else:
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

   # Plot confusion matrix
    fig_cm = plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(cmap='Blues', colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('confusion_matrix.svg', format='svg', bbox_inches='tight')
    plt.close(fig_cm)

    # Plot ROC curve
    fig_roc = plt.figure(figsize=(6, 6))
    # if num_classes == 2:  # Binary classification
    #     plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    # else:  # Multiclass classification
    #     for i in range(num_classes):
    #         plt.plot(fpr[i], tpr[i], label=f"{classes[i]} (AUC = {roc_auc[i]:.2f})")
    
    # plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve")
    # plt.legend(loc="lower right")
    # plt.tight_layout()
    # fig_roc.savefig('roc_curve.svg', format='svg', bbox_inches='tight')
    # plt.close(fig_roc)
    # Return the figure
    return fig_roc, fig_cm

def plot_feature_importance(classifier, feature_names):
    fig, ax = plt.subplots(figsize=(12,6))
    feature_importances = pd.Series(classifier.feature_importances_, index=feature_names)
    feature_importances = feature_importances.sort_values(ascending=False)
    plt.barh(
        feature_importances.index,
        feature_importances.values,
        color="skyblue",
        edgecolor="black",
        height=0.8
    )
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title('Feature Importances from Random Forest')
    plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    return fig


def classify(df, method, splits):
    X_train, X_test, y_train, y_test = prepare_data(df, splits)
    print(len(X_train), len(X_test))
    if method == "Random Forest":
        classifier = RandomForestClassifier(random_state=42)
    elif method == "SVM":
        # for svc and logreg, we need to scale the data
        classifier = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=42))
    elif method == "Logistic Regression":
        classifier = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000))
    
    # y_score is the probability of the sample for each class in the model
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    fig1 = plot_confusion_matrix_roc(y_test, y_score,y_pred)

    if method == "Random Forest":
        fig2 = plot_feature_importance(classifier, X_train.columns)
        return fig1, accuracy_score(y_test, y_pred), fig2
    else:
        return fig1, accuracy_score(y_test, y_pred), None
    

if __name__ == "__main__":
    df = pd.read_csv("./cyto_merged_data.csv")
    x_columns = ["nadh_a1_mean", "nadh_t1_mean","nadh_t2_mean","nadh_tau_mean_mean","fad_a1_mean", "fad_t1_mean","fad_t2_mean","fad_tau_mean_mean"]
    y_column = "cell_type"
    df = df[x_columns + [y_column]]
    splits = 0.7
    # rename x_columns to remove _mean  
    df.columns = [x.replace("_mean", "") for x in df.columns]
    fig, accuracy, fig2 = classify(df, "Random Forest", splits)
    print(accuracy)
    
