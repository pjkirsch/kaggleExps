-- Include required libraries

require 'scripts/funcs'

require 'torch' -- torch

-- Open training data file
local trainFile = assert(io.open("data/mini-train.csv", "r"))
print("Training data file opened.")

-- Initialize variables containing data
nbTrain = 200

trainData = {
	data = torch.Tensor(nbTrain, 28, 28),
	labels = {},
	size = function() return nbTrain end
}

local imageId = 0

print("Reading file...")
for line in trainFile:lines() do
	local i = 1
	local j = 0 
	if imageId ~= 0 then	-- First line containing header is skipped
		for k,pixel in pairs(split(line, ",")) do
			--print(imageId, i, j)
			if j == 0 then
				-- The case i=1, j=0 corresponds to the label info
				trainData.labels[imageId] = tonumber(pixel)
			else
				-- Save the pixel value 
				trainData.data[{imageId, i, j}] = tonumber(pixel)
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
