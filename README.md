# SVM and ANN Model Comparison

This project demonstrates a comparison between Support Vector Machine (SVM) classifiers and Artificial Neural Networks (ANN) using a dataset of hazelnuts with various features.

## Features of the Project

- **Data Preprocessing**:
  - Handles features such as length, width, thickness, and other physical attributes of hazelnuts.
  - Uses normalization and encoding to prepare data for machine learning models.

- **SVM Models**:
  - Compares different kernels (e.g., linear, polynomial, radial basis function (RBF)).
  - Evaluates model performance using metrics such as accuracy and confusion matrices.

- **ANN Models**:
  - Implements a feedforward neural network.
  - Uses backpropagation for training and evaluation.

- **Visualization**:
  - Provides plots for model performance, confusion matrices, and feature importance.

## Dataset

The dataset (`hazelnuts.txt`) includes the following features:

- Length
- Width
- Thickness
- Surface Area
- Mass
- Compactness
- Hardness
- Shell Top Radius
- Water Content
- Carbohydrate Content

The target variable is the hazelnut variety.

## Requirements

To run this project, ensure you have the following libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install the dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. Clone this repository and navigate to the project directory.

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Place the dataset (`hazelnuts.txt`) in the project directory.

3. Open the Jupyter Notebook `SVM_ANN.ipynb`.

4. Follow the steps in the notebook to:
   - Preprocess the dataset.
   - Train and evaluate SVM models with various kernels.
   - Train and evaluate the ANN model.
   - Compare their performance.

## Example Output

### SVM with RBF Kernel
Accuracy: 95%

### ANN Performance
Accuracy: 92%

### Confusion Matrix
![Confusion Matrix](example_confusion_matrix.png)

## Notes

- Adjust hyperparameters to improve the performance of the SVM and ANN models.
- Ensure the dataset is correctly formatted and placed in the appropriate directory.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- `scikit-learn` for machine learning models and utilities.
- `matplotlib` and `seaborn` for visualization.

Feel free to contribute or report issues if you encounter any problems!