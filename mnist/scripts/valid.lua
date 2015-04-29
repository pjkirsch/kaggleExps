-- This script is mostly the one provided by Clement Farabet for Torch tutorial

----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the validation data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------

-- Log results to files
validLogger = optim.Logger(paths.concat(opt.save, 'valid.log'))


print '==> defining validation procedure'

-- validation function
function valid()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over validation data
   print('==> testing on valid set:')
   for t = 1,validData:size() do
      -- disp progress
      xlua.progress(t, validData:size())

      -- get new sample
      local input = validData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = num2classId(validData.labels[t])

      -- valid sample
      local pred = model:forward(input)
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / validData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   validLogger:add{['% mean class accuracy (valid set)'] = confusion.totalValid * 100}
   if opt.plot then
      validLogger:style{['% mean class accuracy (valid set)'] = '-'}
      validLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end
