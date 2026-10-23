# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model to predict a student's final marks based on the number of hours they study

## THEORY
In educational institutions, understanding the direct impact of study habits on academic performance is crucial. The goal of this project is to build a predictive system that takes a single continuous feature—the number of hours a student spends studying—and outputs a continuous numerical prediction of their expected final exam score. Because the target variable (marks) is a continuous numerical value rather than a discrete category, this challenge is classified as a regression problem
## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .


### Name: Raviprasath K

### Register Number: 212224230225


## PROGRAM

```

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Reproducible results
torch.manual_seed(33)

# Sample regression data: learn y = x^2
xt1 = torch.linspace(-2, 2, 100).reshape(-1, 1)
yt1 = xt1 ** 2

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


# Create the model, loss function, and optimizer
plot_model = NeuralNet()
criterion = nn.MSELoss()
plot_optimizer = optim.Adam(plot_model.parameters(), lr=0.001)

# Train the model and record the loss
loss_list = []

for epoch in range(1000):
    plot_optimizer.zero_grad()

    outputs = plot_model(xt1)
    plot_loss = criterion(outputs, yt1)

    plot_loss.backward()
    plot_optimizer.step()

    loss_list.append(plot_loss.item())

# Plot the training loss
plt.figure(figsize=(8, 4))
plt.plot(loss_list)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

```

### Dataset Information

<img width="388" height="696" alt="image" src="https://github.com/user-attachments/assets/b56cf15e-f2b2-440a-a2d4-1ddc278cf5ba" />


### OUTPUT

## Plot Diagram 

<img width="790" height="390" alt="c99c6cd4-f4a6-4937-bf63-a27f1f28d953" src="https://github.com/user-attachments/assets/e219f646-4176-4d34-9f10-4721f84b5338" />


### New Sample Data Prediction

<img width="591" height="162" alt="image" src="https://github.com/user-attachments/assets/26d1bf35-3014-4a86-ad30-056c358552ca" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
