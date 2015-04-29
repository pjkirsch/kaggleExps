-- Include required libraries

require 'scripts/funcs'

require 'torch' -- torch

-- Open training data file
local trainFile = assert(io.open("data/train.csv", "r"))
print("Training data file opened.")

-- Initialize variables containing data
nbTrain = 38000
nbValid = 4000

trainData = {
	data = torch.Tensor(nbTrain, 28, 28),
	labels = {},
	size = function() return nbTrain end
}
validData = {
	data = torch.Tensor(nbValid, 28, 28),
	labels = {},
	size = function() return nbValid end
}
local imageId = 0

print("Reading file...")
for line in trainFile:lines() do
	local i = 1
	local j = 0 
	if imageId ~= 0 then	-- First line containing header is skipped
		for k,pixel in pairs(split(line, ",")) do
			if imageId <= nbTrain then -- Image used as training data
				--print(imageId, i, j)
				if j == 0 then
					-- The case i=1, j=0 corresponds to the label info
					trainData.labels[imageId] = tonumber(pixel)
				else
					-- Save the pixel value 
					trainData.data[{imageId, i, j}] = tonumber(pixel)
				end
			else	-- Image used as validation data
				--print(imageId, i, j)
				if j == 0 then
					-- The case i=1, j=0 corresponds to the label info
					validData.labels[imageId-nbTrain] = tonumber(pixel)
				else
					-- Save the pixel value 
					validData.data[{imageId-nbTrain, i, j}] = tonumber(pixel)
				end

			end
				-- Update indexes
			if j == 28 then
				i = i+1
				j = 1 
			else
				j = j+1
			end	
		end
	end
	-- Increment image Id
	imageId = imageId+1
end

trainFile:close()
print("File closed")
