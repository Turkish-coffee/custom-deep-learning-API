import numpy as np
import matplotlib.pyplot as plt
from Layers import Layer_Dense
from ActivationFunctions import Activation_ReLU, Activation_Sigmoid
from Loss import Activation_Softmax_Loss_CategoricalCrossentropy
from Optimizers import Optimizer_Adam
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=2, n_classes=3, n_clusters_per_class=1, n_informative=2, n_redundant=0, n_repeated=0, random_state=1)

X_test, y_test = X[900:1000], y[900:1000] #load data in those variables
X, y = X[:900], y[:900]

plt.scatter(X[:,0], X[:,1])
plt.show()

plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
plt.show()


# Create Dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 32)
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_Sigmoid()
# Create second Dense layer with 64 input features (as we take output 
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(32, 32)

activation2 = Activation_Sigmoid()
dense3 = Layer_Dense(32,3)
# Create Softmax activation (to be used with Dense layer):

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer
#optimizer = Optimizer_SGD(decay=1e-3,momentum=0.83)
#optimizer = Optimizer_Adagrad(decay=1e-4)
#optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)


# Train in loop
for epoch in range(10001):

    # Make a forward pass of our training data through this layer
    dense1.forward(X)
    # Make a forward pass through activation function
    # it takes the output of first dense layer here
    activation1.forward(dense1.output)
    # Make a forward pass through second Dense layer
    # it takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)
    
    activation2.forward(dense2.output)

    dense3.forward(activation2.output)
    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss 
    loss = loss_activation.forward(dense3.output, y)

    
    # Calculate accuracy from output of activation2 and targets 
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 400:
        print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' + 
            f'loss: {loss:.3f}' +
            f'lr: {optimizer.current_learning_rate}')
    

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense3.backward(loss_activation.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

    # Print gradients
#print(dense1.dweights) 
#print(dense1.dbiases)
#print(dense2.dweights)
#print(dense2.dbiases)

# Validate the model
# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

activation2.forward(dense2.output)

dense3.forward(activation2.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense3.output, y_test)
# Calculate accuracy from output of activation2 and targets # calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1) 
accuracy = np.mean(predictions == y_test)
print(f'validation, acc: {accuracy:.3f},loss: {loss:.3f}')