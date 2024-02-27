# ProjectFeedbackNetworks

This repository includes the implementation of the student project work "Feedback and Forward networks for object recognition
on the data with different noise types" at the University Ulm. The information about the project is provided at the [report]([doc:linking-to-pages#anchor-links](https://github.com/Tmozovka/ProjectFeedbackNetworks/blob/master/Report_Feedback_and_Forward_networks_for_object_recognition%20(1).pdf)).

## Usage

The training of models was executed on the BwUniCluster using Slurm jobs on 4 GPUs. 
The code of the batch jobs can be found in folder batch_jobs. 

The code for training forward VGG network is provided in "batch_jobs\Train forward VGG16 model\train_forward_model.py".

The code for training procedure for Feedback models can be found in "batch_jobs/experiment_frozen_network/training_procedure.py" .
Based on the training procedure different types of Feedback models are trained. 
The code for training procedure on Gaussian Noise can be found in "batch_jobs/experiment_frozen_network_train_gaussian_noise/training_procedure.py" .
The training of not frozen Feedback model with connection from block 4 to block 1 is provided in file: "batch_jobs\experiment_not_frozen_networks\feedback\1. experiment\4block_to_1block.py" .

The data preparation was performed inside the Jupiter Notebook and can be found in folder "notebooks/Prepare data/".

The evaluation of the results is executed in following Jupiter Notebooks: "notebooks/Visualise accuracies.ipynb" and "Notebooks notebooks/Visualise evaluation.ipynb" .

The training results can be found in folder reports/ .

The architecture of the trained models is provided in src/models .

