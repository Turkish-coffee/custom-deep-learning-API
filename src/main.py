import numpy as np
import matplotlib.pyplot as plt
from Layers import Layer_Dense, Layer_Dropout
from ActivationFunctions import Activation_ReLU, Activation_Sigmoid, Activation_Linear, Activation_Softmax
from Loss import Activation_Softmax_Loss_CategoricalCrossentropy, Loss_MeanSquaredError, Loss_CategoricalCrossentropy
from Optimizers import Optimizer_Adam
from sklearn.datasets import make_classification, make_regression
from Metrics import Accuracy_Regression, Accuracy_Categorical
from Model import Model

X, y = make_classification(n_samples=1000, n_features=2, n_classes=3, n_clusters_per_class=1, n_informative=2, n_redundant=0, n_repeated=0, random_state=1)

X_test, y_test = X[900:1000], y[900:1000] #load data in those variables
X, y = X[:900], y[:900]

plt.scatter(X[:,0], X[:,1])
plt.show()

plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
plt.show()


# Instantiate the model
model = Model()
# Add layers
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4,
bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())
# Set loss, optimizer and accuracy objects
model.set(
loss=Loss_CategoricalCrossentropy(),
optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
accuracy=Accuracy_Categorical()
)
# Finalize the model
model.finalize()
# Train the model
model.train(X, y, validation_data=(X_test, y_test),
epochs=10000, print_every=100)