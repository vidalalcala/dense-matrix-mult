----------------------------------------------------------------------
-- A test for the OpenBLAS library with Torch7 tensors
--
-- (Jose V. Alcala-Burgos, 2014)
--

require 'torch'
require 'pl'

function torch.gercol(v,w)
   local n = v:size(1)
   local m = w:size(1)
   local vw = torch.Tensor( n, m)
   for j =  1, m, 1 do
      local vwj = vw:select(2,j)
      vwj = torch.mul(v , w[j])
   end
   return vw
end

-- Parse command-line options
local opt = lapp([[
   -t,--threads       (default 11)          number of threads
   -N,--numRows		  (default 10100) 		number of matrix rows
   -i,--iterations   (default 100)         number of iterations
]])

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- create matrices
N = opt.numRows
A = torch.Tensor(N,N)
v = torch.Tensor(N)
w = torch.Tensor(N)

-- start timer
timer = torch.Timer()
print('<torch> timer started')

-- perform matrix operation

-- Find botlenecks

R = torch.Tensor(N,N)

luatrace = require("luatrace")
luatrace.tron()

for j = 1 , opt.iterations , 1 do
   print('<torch> Iteration: ', j , ' of ', opt.iterations)
   R = torch.ger(v,w)
end

luatrace.troff()

-- show elapsed time
print('Time elapsed for ' , opt.iterations, ' multiplications of two matrices with ' .. opt.numRows .. ' rows : ' .. timer:time().real .. ' seconds')

