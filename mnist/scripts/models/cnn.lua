-- Define a neural network model with 3 Convolutional layers + 2 fully-connected layers + 1 softmax layer

-- Load libraries
require 'torch'
require 'nn'

-- Define parameters
height = 28
width = 28
nPlanes = {1,16,16,16}
poolRegion = 3
poolStep = 2

nHL1 = 300
nHL2 = 100
nOutputs = 10

-- Define model
model = nn.Sequential() -- The NN model is considered as a sequence of functions
model:add(nn.SpatialConvolutionMM(nPlanes[1], nPlanes[2], 5, 5)) -- 16 filters 1*5*5
model:add(nn.Tanh()) -- Hyperbolic tangent non-linear function
model:add(nn.SpatialMaxPooling(poolRegion, poolRegion, poolStep, poolStep))

model:add(nn.SpatialConvolutionMM(nPlanes[2], nPlanes[3], 3, 3)) -- 16 filters 16*3*3
model:add(nn.Tanh()) -- Hyperbolic tangent non-linear function

model:add(nn.SpatialConvolutionMM(nPlanes[3], nPlanes[4], 3, 3)) -- 16 filters 16*3*3
model:add(nn.Tanh()) -- Hyperbolic tangent non-linear function

model:add(nn.Reshape(nPlanes[4]*7*7))
model:add(nn.Linear(nPlanes[4]*7*7, nHL1)) -- Connection weights
model:add(nn.Tanh()) -- Hyperbolic tangent non-linear function

model:add(nn.Linear(nHL1, nHL2)) -- Connection weights
model:add(nn.Tanh()) -- Hyperbolic tangent non-linear function

model:add(nn.Linear(nHL2, nOutputs)) -- Connection weights
model:add(nn.LogSoftMax())

-- Display model description
print("Model description:")
print(model)

-- Define cost function
cfn = nn.ClassNLLCriterion()
print("Cost function:")
print(cfn)
