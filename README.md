### Neural Networks Regression in TensorFlow

### Project Description:
This project focuses on building, training, and evaluating deep learning neural network regression models using TensorFlow 2.x.

The main objective is to understand the mechanics of deep learning regression—predicting a continuous numeric output given a set of features.

It covers foundational concepts such as input/output shapes, optimizing models by adjusting hyperparameters (layers, hidden units, learning rates), visualizing predictions, and preprocessing features with normalization.


### Problem Statement:
The project addresses two regression problems:

Synthetic Data Linear Mapping: Learning the simple underlying linear relationship ($y = X + 10$) from a minimal custom dataset.

Medical Insurance Premium Prediction: Building a deep learning model to predict individual medical insurance costs based on demographic data and lifestyle attributes (age, sex, BMI, children, smoker status, and region).


### Dataset Used:
**Synthetic Dataset**: A toy dataset containing 8 feature points (X) and 8 corresponding labels (y) generated using NumPy arrays.

**Medical Insurance Dataset**: A real-world dataset loaded via a public web URL. It contains demographic variables used to estimate healthcare charges.

**Features**: * Numerical: age, bmi, children

**Categorical (Encoded via One-Hot Encoding)**: sex, smoker, region

**Target Variable**: charges (continuous numerical values)


### Model Architecture:
The project iteratively builds and improves various model structures. The final and most optimized model (insurance_model_4) utilizes the following deep neural network structure:

**Input Layer**: Accepts preprocessed and scaled features (x_train_normal).

**Hidden Layer 1**: Fully connected (Dense) layer with 100 hidden units and ReLU activation.

**Hidden Layer 2**: Fully connected (Dense) layer with 10 hidden units and ReLU activation.

**Output Layer**: A single Dense neuron to output the continuous numerical prediction.

### Compilation Details:
**Loss Function**: Mean Absolute Error (MAE)

**Optimizer**: Adam

**Evaluation Metric**: MAE


### Results:
The medical insurance model was optimized across 4 main stages:

**insurance_model_1**: Baseline model with 2 dense layers trained for 100 epochs using the SGD optimizer.

**insurance_model_2**: Increased capacity (Dense layers with 100 and 1 neurons) and changed to the Adam optimizer, resulting in a significantly reduced MAE.

**insurance_model_3**: Kept the model structure identical to model 2 but trained for longer (300 epochs).

**insurance_model_4 (Final Model)**: Built with two hidden layers (100 and 10 units), trained for 300 epochs, and fed with min-max normalized data (x_train_normal)

**using Scikit-Learn's MinMaxScaler**. Feature normalization alongside the complex architecture yielded the lowest test loss and standard error.

### How to Run
Clone this repository onto your workstation.

Install deep learning and analytical tabular processing packages: **pip install tensorflow numpy pandas matplotlib scikit-learn**

Launch your local notebook infrastructure and open **01_neural_network_regression_with_tensorflow.ipynb**.

Execute cells sequentially to create synthetic matrices, ingest the tabular insurance files, trigger min-max scaling transformations, and map out the training progress charts.

### Future Improvements
Integrate early-stopping parameters **(EarlyStopping)** to automatically halt training weights the moment evaluation improvements stall.

Experiment with deeper topologies or higher neuron densities to observe performance trade-offs on complex non-linear numeric inputs.

Scale up the dataset loading mechanics to process multi-output regression pipelines simultaneously.
