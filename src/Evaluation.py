import os
import numpy as np
from matplotlib import pyplot as plt
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
    
def run_mcnemar_test(model_1, model_2, test_ds, path_to_persist=None):
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
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')
        
    if path_to_persist:
        result = {}
        

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
        for idx in range(len(X_test[0])):
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
        

    