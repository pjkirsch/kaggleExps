-- Include required libraries

require 'scripts/funcs'

require 'torch' -- torch

-- Open testing data file
local testFile = assert(io.open("data/test.csv", "r"))
print("Testing data file opened.")

-- Initialize variables containing data
testData = torch.Tensor(28000, 28, 28)

local imageId = 0

print("Reading file...")
for line in testFile:lines() do
	local i = 1
	local j = 1 
	if imageId ~= 0 then	-- First line containing header is skipped
		for k,pixel in pairs(split(line, ",")) do
			--print(imageId, i, j)
			
			-- Save the pixel value 
			testData[{imageId, i, j}] = tonumber(pixel)
			
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

testFile:close()
print("File closed")