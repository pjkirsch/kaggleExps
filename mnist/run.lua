require "scripts/funcs"

dofile("scripts/loadTrain.lua")
dofile("scripts/loadTest.lua")

dofile("scripts/models/cnn-ReLU.lua")

dofile("scripts/train.lua")
dofile("scripts/valid.lua")
dofile("scripts/test.lua")

while true do
	train()
	valid()
	test()
end
