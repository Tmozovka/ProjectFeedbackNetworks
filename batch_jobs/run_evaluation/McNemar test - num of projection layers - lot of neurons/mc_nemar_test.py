import sys
sys.path.append('../../../')
from src.models.FeedbackModelBuilder import VGG16FeedbackFrozen4BlockTo4Block,\
                                            VGG16FeedbackFrozen5BlockTo5Block,\
                                            VGG16FeedbackFrozen4BlockTo1Block,\
                                            VGG16FeedbackFrozen5BlockTo3Block,\
                                            VGG16FeedbackFrozen5BlockTo4Block
from src.models.FeedbackModelBuilder import VGG16FeedbackFrozen5BlockTo3Block3ProjLayers, VGG16FeedbackFrozen5BlockTo3Block2ProjLayers
from src.models.ForwardModelBuilder import VGG16CustomFrozen
from src.Evaluation import run_mcnemar_test, persist_mcnemar_test
from src.DatasetNoise import tf_gaussian_noise, tf_salt_pepper_noise
from PyPDF2 import PdfMerger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import json
import os
import io

path_to_datasets = '../../../../../experiments/create_datasets/normalize_dataset/saved_datasets/'


batch_size = 256
num_classes = 16
img_size = 224
input_shape = (None, img_size, img_size, 3)


name_model_path = {
"frozenVGG16-Feedback_5block_to_3block": {
    "model_class": VGG16FeedbackFrozen5BlockTo3Block,
    "checkpoint_filepath":"../../../models/frozenVGG16/Feedback_5block_to_3block/checkpoint"
                                         },
"frozenVGG16-Feedback_4block_to_4block_3_projection_layers": {
    "model_class": VGG16FeedbackFrozen5BlockTo3Block3ProjLayers,
    "checkpoint_filepath":"../../../models/frozenVGG16/Feedback_5block_to_3block_3_projection_layers/checkpoint"
                                         },
"frozenVGG16-Feedback_4block_to_1block_2_projection_layers": {
    "model_class": VGG16FeedbackFrozen5BlockTo3Block2ProjLayers,
    "checkpoint_filepath":"../../../models/frozenVGG16/Feedback_5block_to_3block_2_projection_layers/checkpoint"
                                         }
}

def build_model(model_class, model_checkpoint_filepath):
    sample = next(iter(test_ds))[0]
    model = model_class()
    model.build(input_shape)
    model(sample)
    model.summary(show_trainable=True)
    model.load_weights(model_checkpoint_filepath)
    return model

if __name__ == "__main__":
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    all_result = dict()
    with strategy.scope():
        test_ds_prepared_without_batch = tf.data.Dataset.load(os.path.join(path_to_datasets, "test_ds_without_batching"))
        test_ds = test_ds_prepared_without_batch.batch(batch_size).prefetch(4)
        test_ds_gaussian_noise = test_ds_prepared_without_batch.map(tf_gaussian_noise).batch(batch_size).prefetch(4)
        test_ds_salt_pepper_noise = test_ds_prepared_without_batch.map(tf_salt_pepper_noise).batch(batch_size).prefetch(4)
        len_ds = len(test_ds_prepared_without_batch)
        
        path_to_persist = "../../../reports/Mc_Nemar_Test_diff_num_proj_layers/5_to_3_block/"
        
        
        
        for test_ds, name_test_ds in zip([test_ds, test_ds_gaussian_noise, test_ds_salt_pepper_noise], \
                                             ['original_data', 'gaussian_noise', 'salt_and_pepper_noise']):
            merger = PdfMerger()
            curr_path_to_persist_results = os.path.join(path_to_persist, name_test_ds)
            all_result[name_test_ds] = dict()
            for model_1, model_2 in itertools.combinations([(k,v) for k,v in name_model_path.items()], 2):
                print(model_1)
                print(model_2)
                model_1_class = model_1[1]["model_class"]
                model_1_checkpoint_filepath = model_1[1]["checkpoint_filepath"]
                model_1_name = model_1[0]

                model_2_class = model_2[1]["model_class"]
                model_2_checkpoint_filepath = model_2[1]["checkpoint_filepath"]
                model_2_name = model_2[0]

                model_1 = build_model(model_1_class, model_1_checkpoint_filepath)
                model_2 = build_model(model_2_class, model_2_checkpoint_filepath)

                result = run_mcnemar_test(model_1, model_2, test_ds, model_1_name=model_1_name, model_2_name=model_2_name)
                all_result[name_test_ds][f"{model_1_name}_{model_2_name}"] = result
                print(result)

                folder_name = "significant_difference" if result["models_diff"] else "not_significant_difference"
                path_to_persist_results = os.path.join(curr_path_to_persist_results, folder_name, f"{model_1_name}_{model_2_name}")
                if not os.path.exists(path_to_persist_results):
                    os.makedirs(path_to_persist_results)

                path_file_test = os.path.join(path_to_persist_results, "mc_nemar_test.json")
                path_png_test = os.path.join(path_to_persist_results, 'mc_nemar_test_result.png')
                if os.path.exists(path_file_test):
                    os.remove(path_file_test)
                with open(path_file_test, "a") as write_file:
                    json.dump(result, write_file, indent=4)

                df = pd.DataFrame(result['contingency_table'],\
                                  columns=['Model 2 correct', 'Model 2 incorrect'],\
                                  index=['Model 1 correct', 'Model 1 incorrect'])



                ax = sns.heatmap(df, annot=True, fmt=".0f", vmin=0, vmax=len_ds)
                ax.set(xlabel="", ylabel="", title=f"McNemar test between:\n Model 1: {model_1_name} \n Model 2: {model_2_name} \n Result: {result['evaluation_result']}\n")
                ax.xaxis.tick_top()
                ax.figure.set_tight_layout(True)
                plt.savefig(path_png_test)
                pdf_buffer = io.BytesIO()
                plt.savefig(pdf_buffer, format='pdf')
                merger.append(pdf_buffer)
                plt.close()
                pdf_buffer.close()
                print("--------------------------------------------------")

            #persist_mcnemar_test(result, path_to_persist, heatmap_max=len_ds, model_1_name = model_1_name, model_2_name = model_2_name)
            merger.write(os.path.join(curr_path_to_persist_results, "result.pdf"))
            merger.close()
        
        path_file_all_result = os.path.join(path_to_persist, "mc_nemar_test.json")
        if os.path.exists(path_file_all_result):
                os.remove(path_file_all_result)
        with open(path_file_all_result, "a") as write_file:
                json.dump(all_result, write_file, indent=4)
            