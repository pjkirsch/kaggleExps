-- Include required libraries

require 'scripts/funcs'

require 'torch' -- torch
require 'xlua'	-- for progression bar

-- Open testing data file
local testFile = assert(io.open("data/test.csv", "r"))
print("Testing data file opened.")

-- Initialize variables containing data
nbTest = 28000

testData = {
	data = torch.Tensor(nbTest, 28, 28),
	size = function() return nbTest end
}
local imageId = 0

print("Reading file...")
for line in testFile:lines() do
	local i = 1
	local j = 1 
	-- Display progression		
	xlua.progress(imageId, 28000)

	if imageId ~= 0 then	-- First line containing header is skipped
		for k,pixel in pairs(split(line, ",")) do
			--print(imageId, i, j)
			
			-- Save the pixel value 
			testData.data[{imageId, i, j}] = rescalePixFeat(tonumber(pixel))
			
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
