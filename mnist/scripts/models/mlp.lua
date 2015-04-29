-- Define a neural network model with 1 hidden layer and 1 softmax layer

-- Load libraries
require 'torch'
require 'nn'

-- Define parameters
height = 28
width = 28

nInputs = height*width
nHL1 = 100
nOutputs = 10

-- Define model
model = nn.Sequential() -- The NN model is considered as a sequence of functions
model:add(nn.Reshape(nInputs)) -- Reshape features from 28*28 to 784*1
model:add(nn.Linear(nInputs, nHL1)) -- Connection weights
model:add(nn.Tanh()) -- Hyperbolic tangent non-linear function
model:add(nn.Linear(nHL1, nOutputs)) -- Connection weights
model:add(nn.LogSoftMax())

-- Display model description
print("Model description:")
print(model)

-- Define cost function
cfn = nn.ClassNLLCriterion()
print("Cost function:")
print(cfn)
