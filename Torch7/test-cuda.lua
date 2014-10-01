----------------------------------------------------------------------
-- A test for the OpenBLAS library with Torch7 tensors
--
-- (Jose V. Alcala-Burgos, 2014)
--

require 'torch'
require 'pl'
require 'cutorch'
print(  cutorch.getDeviceProperties(cutorch.getDevice()) )


function torch.gercuda(v,w)
   
   local n = v:size(1)
   local m = w:size(1)
   local vw = torch.CudaTensor( n, m)
   for j =  1, m, 1 do
      wj = torch.CudaTensor(1)
      wj:float()
      local vwj = vw:select(2,j)
      v:cuda()
      vwj:mul( v , wj )
   end
   return vw
end

-- Parse command-line options
local opt = lapp([[
   -t,--threads       (default 2)          number of threads
   -N,--numRows		  (default 10002) 		number of matrix rows
   -i,--iterations   (default 10000)         number of iterations
]])

-- threads
--torch.setnumthreads(opt.threads)
--print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats
-- torch.setdefaulttensortype('torch.FloatTensor')

-- start timer
timer = torch.Timer()
print('<torch> timer started')


-- create matrices
N = opt.numRows
A = torch.CudaTensor( N , N )
v = torch.CudaTensor( N )
w = torch.CudaTensor( N )

-- perform matrix operation

--luatrace = require("luatrace")
--luatrace.tron()

for j = 1 , opt.iterations , 1 do
   print('<torch> Iteration: ', j , ' of ', opt.iterations)

   w:addmv( A , v )
end

--luatrace.troff()

-- show elapsed time
print('Time elapsed for ' , opt.iterations, ' matrix-vector multiplications with ' .. opt.numRows .. ' rows : ' .. timer:time().real .. ' seconds')

