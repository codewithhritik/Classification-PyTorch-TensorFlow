# Multi-Platform Classification with TensorFlow and PyTorch

This repository contains code for performing multi-class classification tasks on two different datasets using TensorFlow and PyTorch. The datasets used are:

1. **Diabetes Health Indicators Dataset**: This dataset contains healthcare statistics, lifestyle survey information, and the diagnosis of diabetes for each patient. The goal is to classify patients as diabetic, pre-diabetic, or healthy based on the given features.

2. **Human Activity Recognition (HAR70+) Dataset**: This dataset contains accelerometer data collected from older adults (70-95 years old) performing various activities. The goal is to classify the activities based on the accelerometer readings.

## Major Steps

### Diabetes Health Indicators Dataset

**TensorFlow:**

1. Load the dataset and preprocess the data (fill missing values, one-hot encode categorical features, scale numerical features)
2. Split the data into train, validation, and test sets
3. Define the model architecture (4 layers with ReLU activations)
4. Compile the model (optimizer=Adam(lr=0.0001), loss=binary_crossentropy, metrics=[accuracy])
5. Train the model for 30 epochs with batch_size=1024 and 20% validation split
6. Evaluate the model on the test set and calculate performance metrics
7. Apply a custom threshold of 0.65 and calculate accuracy
8. Plot ROC, Precision-Recall curve, predicted probability histogram, training vs validation loss, and test loss for every 20th batch

**PyTorch:**

1. Load the dataset and preprocess the data (fill missing values, one-hot encode categorical features, scale numerical features)
2. Split the data into train, validation, and test sets, and create DataLoaders
3. Define the model architecture (6 layers with ReLU, BatchNorm, and Dropout)
4. Set the loss function as BCELoss and the optimizer as Adam(lr=0.001)
5. Train the model for 30 epochs
6. Evaluate the model on the test set and calculate performance metrics
7. Apply a custom threshold of 0.65 and calculate accuracy
8. Plot ROC, Precision-Recall curve, predicted probability histogram, training vs validation loss, and test loss for every 20th batch

### Human Activity Recognition (HAR70+) Dataset

**TensorFlow:**

1. Load the dataset and preprocess the data (convert timestamp, adjust labels, scale features)
2. Split the data into train, validation (10%), and test (20%) sets
3. Define the model architecture (2 Dense layers with 64 neurons, Dropout, output Softmax for 8 classes)
4. Compile the model (optimizer=adam, loss=sparse_categorical_crossentropy, metrics=[accuracy])
5. Train the model with EarlyStopping callback for 20 epochs with 10% validation split
6. Evaluate the model on the test set and calculate performance metrics
7. Save the trained model
8. Plot training vs validation loss and test loss for every 20th batch (up to 800 batches)

**PyTorch:**

1. Load the dataset and preprocess the data (convert timestamp, adjust labels, scale features)
2. Split the data into train, validation (10%), and test (20%) sets, and create DataLoaders
3. Define the model architecture (3-layer MLP with ReLU and Dropout)
4. Set the loss function as CrossEntropyLoss and the optimizer as Adam(lr=0.001)
5. Train the model for 20 epochs
6. Evaluate the model on the test set and calculate performance metrics
7. Plot training vs validation loss and test loss for every 20th batch

## Results and Performance Comparison

The performance of the models on the test sets is summarized below:

### Diabetes Health Indicators Dataset

| Platform   | Test Accuracy | Test Loss |
| ---------- | ------------- | --------- |
| TensorFlow | 0.8681        | 0.3086    |
| PyTorch    | 0.8667        | 0.3109    |

### Human Activity Recognition (HAR70+) Dataset

| Platform   | Test Accuracy | Validation loss after 20th Epoch |
| ---------- | ------------- | -------------------------------- |
| TensorFlow | 96.52%        | 0.1122                           |
| PyTorch    | 96.54%        | 0.1215                           |

For more details, including plots and comparisons, please refer to the `report.pdf` file.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
