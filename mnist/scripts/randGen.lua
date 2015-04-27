-- Include required libraries
require 'scripts/funcs'

require 'torch' -- torch

-- Load test set
dofile("scripts/loadTest.lua")

-- Open output file
local outputFile = assert(io.open("results/rand.csv", "w"))
print("Output file opened.")

-- Write header
outputFile:write("ImageId,Label\n")

for i = 1,testData:size(1) do
outputFile:write(i .. "," .. 0 .. "\n")
end

outputFile:close()
print("Output file closed.")

