import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow import keras
import sys
import json
#sys.path.append('../')
from statsmodels.stats.contingency_tables import mcnemar


def persist_evaluation_results(accuracy, history, path_to_persist):
    path_file_accuracy = os.path.join(path_to_persist, "accuracies.json")
    path_file_history = os.path.join(path_to_persist, "history.json")
    for di, path in zip([accuracy, history], [path_file_accuracy, path_file_history]):
        if os.path.exists(path):
            os.remove(path)
        with open(path, "a") as write_file:
            json.dump(di, write_file, indent=4)
    plot_model_history(history, path_to_persist=os.path.join(path_to_persist, "train_history.png"))

def plot_model_history(history, path_to_persist=None):
    loss = history['loss']
    val_loss = history['val_loss']

    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    eps = range(len(loss))

    figure, axis = plt.subplots(2, 1)
    axis[0].plot(eps, loss, 'r', label='Training loss')
    axis[0].plot(eps, val_loss, 'b', label='Validation loss')
    axis[0].set_title('Training and Validation Loss')
    axis[0].set_xlabel('Epoch')
    axis[0].set_ylabel('Loss Value')
    axis[0].set_ylim([0, 2])
    axis[0].legend()

    axis[1].plot(eps, accuracy, 'r', label='Training accuracy')
    axis[1].plot(eps, val_accuracy, 'b', label='Validation accuracy')
    axis[1].set_title('Training and Validation accuracy')
    axis[1].set_xlabel('Epoch')
    axis[1].set_ylabel('Accuracy Value')
    axis[1].set_ylim([0, 1])
    axis[1].legend()
    
    figure.tight_layout()
    if path_to_persist:
        figure.savefig(path_to_persist)
        plt.close(figure)
    else:
        print(f'Highest Validation Accuracy: {np.max(val_accuracy)}')
        plt.show()
    
    
def get_confusion_matrix_for_model_and_data(
    model: keras.Model, x_test, y_test
) -> np.ndarray:
    """
    Use the model to predict the classes of the test data and then return the confusion matrix. You can visualize them
    as a matplotlib plot using the visualize_confusion_matrix function.

    Usage:
    Returns a matrix C where C_{i, j} is equal to the number of observations known to be in group i but predicted to be in group j.
    That is if the model predicts the class 0 for 10 observations and the true class is 1 for 5 of them, then C[0, 1] = 5.

    For binary classification, C[0, 0] is the number of true negatives, C[0, 1] is the number of false positives,
    C[1, 0] is the number of false negatives, and C[1, 1] is the number of true positives. A perfect classifier
    would have C[0, 1] = C[1, 0] = 0. More general: In a perfect classifier, all entries of C are zero except for those
    on the main diagonal, which are equal to the number of observations in each group.

    :param model: The model to use for prediction.
    :param x_test: The test data.
    :param y_test: The test labels.
    :return: The confusion matrix as a numpy array.
    """
    y_test_prepared = list()
    y_pred_prepared = list()
    for batch_idx in range(len(x_test)):
        prediction = model.predict(x_test[batch_idx])
        prediction = np.argmax(prediction, axis=1)
        right_prediction = y_test[batch_idx]
        y_test_prepared.extend(right_prediction)
        y_pred_prepared.extend(prediction)
    return confusion_matrix(y_test_prepared, y_pred_prepared)


def visualize_confusion_matrix(
    confusion_matrix_: np.ndarray, model_name: str, dataset_name: str, path_to_save = None
) -> None:
    """
    Visualize the confusion matrix as a heatmap using seaborn and matplotlib.
    Model Name and Dataset Name are used for the title of the plot.

    :param confusion_matrix_: The confusion matrix to visualize generated by the get_confusion_matrix_for_model_and_data function.
    :param model_name: The name of the model.
    :param dataset_name: The name of the dataset.
    """
    plt.figure(figsize=(8, 5))
    sns.heatmap(confusion_matrix_, annot=True, fmt="d")
    plt.title(f"Confusion Matrix for {model_name} on {dataset_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if path_to_save:
        plt.savefig(path_to_save)
        plt.close()
    else:
        plt.show()

    
def run_mcnemar_test(model_1, model_2, test_ds, model_1_name = "", model_2_name = ""):
    """
    Source: https://machinelearningmastery.com/mcnemars-test-for-machine-learning/
    """
            
    # define contingency table
    table = create_contingency_table(model_1, model_2, test_ds)
    # calculate mcnemar test
    result = mcnemar(table, exact=True)
    # summarize the finding
    print("Results McNemar's Test: ")
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    eval_res = ""
    if result.pvalue > alpha:
        eval_res = 'Same proportions of errors (fail to reject H0)'
        models_diff = False
    else:
        eval_res = 'Different proportions of errors (reject H0)'
        models_diff = True
      
    result = {
        "statistics": result.statistic, 
        "p-value": result.pvalue,
        "evaluation_result": eval_res,
        "models_diff":models_diff,
        "contingency_table": table,
        "model_1_name":model_1_name,
        "model_2_name":model_2_name
        }
   
            
    return result


def persist_mcnemar_test(result, path_to_persist, heatmap_max=4000, model_1_name = "", model_2_name = ""):
    path_file_test = os.path.join(path_to_persist, "mc_nemar_test.json")
    path_png_test = os.path.join(path_to_persist, 'mc_nemar_test_result.png')
    if os.path.exists(path_file_test):
        os.remove(path_file_test)
    with open(path_file_test, "a") as write_file:
        json.dump(result, write_file, indent=4)

    df = pd.DataFrame(result['contingency_table'],\
                      columns=['Model 2 correct', 'Model 2 incorrect'],\
                      index=['Model 1 correct', 'Model 1 incorrect'])
    ax = sns.heatmap(df, annot=True, fmt=".0f", vmin=0, vmax=heatmap_max)
    ax.set(xlabel="", ylabel="", title=f"McNemar test between:\n Model 1: {model_1_name} \n Model 2: {model_2_name} \n Result: {result['evaluation_result']}\n")
    ax.xaxis.tick_top()
    ax.figure.set_tight_layout(True)
    plt.savefig(path_png_test)
    

def create_contingency_table(model_1, model_2, test_ds):
    X_test = list(map(lambda x: x[0], test_ds))
    y_test = list(map(lambda x: x[1], test_ds))
    
    correct_1_correct_2 = 0
    correct_1_incorrect_2 = 0
    incorrect_1_incorrect_2 = 0
    incorrect_1_correct_2 = 0
    
    for batch_idx in range(len(X_test)):
        predict_batch_model_1 = model_1.predict(X_test[batch_idx])
        predict_batch_model_2 = model_2.predict(X_test[batch_idx])
        for idx in range(len(predict_batch_model_1)):
            prediction_1 = np.where(predict_batch_model_1[idx] == max(predict_batch_model_1[idx]))
            prediction_2 = np.where(predict_batch_model_2[idx] == max(predict_batch_model_2[idx]))
            right_prediction = y_test[batch_idx][idx]
            model_1_correct = prediction_1 == right_prediction
            model_2_correct = prediction_2 == right_prediction
            if model_1_correct and model_2_correct:
                correct_1_correct_2 += 1
            elif model_1_correct and not model_2_correct:
                correct_1_incorrect_2 += 1
            elif not model_1_correct and model_2_correct:
                incorrect_1_correct_2 += 1
            else:
                incorrect_1_incorrect_2 += 1
            
    return [[correct_1_correct_2, correct_1_incorrect_2],
     [incorrect_1_correct_2, incorrect_1_incorrect_2]]
        

    