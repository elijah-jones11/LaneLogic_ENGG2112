# LaneLogic Data & Supporting Code

This GITHUB Repo contains all the supporting code and datasets used for the LaneLogic ENGG2112 project: Here is a brief descrpition on where to find everything:

The main dataset & local dataset videos have been removed as they were ~14 GB each :()



###### Project File Guide: ######

# initial_yolo_model_understanding.py

# YOLO
The original early test script used to understand how YOLOv8 works for detecting objects in traffic footage, as well as scripts used to run YOLOv8 on the recorded traffic footage and export detected object counts into CSV files.

# Cleaned_Datasets_local
Cleaned CSV datasets created from the local YOLOv8 traffic footage outputs.

# figures_local
Figures and graphs generated from the local traffic footage data.

# Datasets_main
Main external traffic dataset used for supervised machine learning model training and analysis from Afghanistan.

# Cleaned_Dataset_main
The cleaned version of the main traffic dataset.

# Traffic_engineered_fixed.csv
Final engineered traffic CSV file used for model training/testing.

# patterns_of_days
Analysis of daily and weekly traffic patterns for our main dataset.

# train_valid_test
Contains all the code for splitting the data into training/validation/testing as well as the grid searching and choosing appropriate hyperparameters + figures.

# only_train_test
Using those optimised hyperparameters and training them on the training + validation data before on the testing data and achieving the results.

# figures
Main figures generated from the final dataset using optimised Random Forest model. 

# green_cycles
Comparing our adaptive model to a simplified fixed cycle model over the testing dataset.

# presentation_demo
Files used to create the presentatino yoloV8 demos.



Thank you - From Group 3!!
