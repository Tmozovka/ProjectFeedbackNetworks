import sys
sys.path.append('../../../')
import tensorflow as tf
import os
import time
import logging
from src.Helper import append_to_file
from src.DatasetNoise import tf_gaussian_noise, tf_salt_pepper_noise
from src.Evaluation import persist_evaluation_results
from src.models.FeedbackModelBuilder import VGG16Feedback4BlockTo1Block, VGG16FeedbackFrozen4BlockTo1Block

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/bwhpc/common/devel/cuda/11.8"


if __name__ == "__main__":

    path_to_datasets = '../../../../../experiments/create_datasets/normalize_dataset/saved_datasets/'
    train_ds_prepared_without_batch = tf.data.Dataset.load(os.path.join(path_to_datasets, "train_ds_without_batching"))
    test_ds_prepared_without_batch = tf.data.Dataset.load(os.path.join(path_to_datasets, "test_ds_without_batching"))
    valid_ds_prepared_without_batch = tf.data.Dataset.load(os.path.join(path_to_datasets, "valid_ds_without_batching"))

    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)

    batch_size = 8
    num_classes = 16
    img_size = 224
    input_shape = (None, img_size, img_size, 3) 
    epochs= 20
    print("batch_size", batch_size)
    print("num epochs", epochs)
    print("input_shape", input_shape)
    
    path_to_persist_results = f"../../../reports/notFrozenVGG16/VGG16Feedback 4Block To 1Block"
    if not os.path.exists(path_to_persist_results):
        os.makedirs(path_to_persist_results)
    checkpoint_filepath = f'../../models/notFrozenVGG16/VGG16Feedback 4Block To 1Block/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    with strategy.scope():
        train_ds = train_ds_prepared_without_batch.shuffle(10000).batch(batch_size).prefetch(4)
        valid_ds = valid_ds_prepared_without_batch.batch(batch_size).prefetch(4)
        test_ds = test_ds_prepared_without_batch.batch(batch_size).prefetch(4)
        test_ds_gaussian_noise = test_ds_prepared_without_batch.map(tf_gaussian_noise).batch(batch_size).prefetch(4)
        test_ds_salt_pepper_noise = test_ds_prepared_without_batch.map(tf_salt_pepper_noise).batch(batch_size).prefetch(4)

        sample = next(iter(test_ds))[0]
        model = VGG16Feedback4BlockTo1Block()
        model.build(input_shape)
        model(sample)
        model.summary(show_trainable=True)
        

        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        start_time = time.time()
        history = model.fit(train_ds, epochs=epochs, validation_data=valid_ds, callbacks=[model_checkpoint_callback], verbose=1) 
        execution_time = time.time() - start_time
        results = model.evaluate(test_ds, batch_size=batch_size)
        results_gaussian_noise = model.evaluate(test_ds_gaussian_noise, batch_size=batch_size)
        results_salt_pepper_noise = model.evaluate(test_ds_salt_pepper_noise, batch_size=batch_size)

    test_accuracies = {
        "test_acc_original_data": results[1], 
        "test_acc_gaussian_noise": results_gaussian_noise[1],
        "test_acc_salt_pepper_noise": results_salt_pepper_noise[1]
        }
        
    persist_evaluation_results(accuracy=test_accuracies, history=history.history, path_to_persist=path_to_persist_results)  

    

