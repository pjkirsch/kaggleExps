-- This script is mostly based on  the one provided by Clement Farabet for Torch tutorial (test for supervised case)

----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

require 'scripts/funcs'
----------------------------------------------------------------------

print '==> defining test procedure'


-- test function
function test()
	-- Open file containing results
	local resultFile = assert(io.open("results/hyp_epoch".. epoch .. ".csv", "w"))
	print("Test results file opened.")

	-- write file header
	resultFile:write("ImageId,Label\n")

   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end

      -- test sample
      local _,pred = torch.max(model:forward(input),1) -- return the predicted class
		resultFile:write(t, ',', classId2num(pred[1]),"\n") -- Write the predicted number
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
	-- Close test results file
	resultFile:close()
	print("Test results file closed.")
	
   -- next iteration:
   confusion:zero()
end
