----------------------------------------------------------------------
-- A test for the OpenBLAS library with Torch7 tensors
--
-- (Jose V. Alcala-Burgos, 2014)
--

require 'torch'
require 'pl'

-- Parse command-line options
local opt = lapp([[
   -t,--threads       (default 8)           number of threads
   -N,-numRows		  (default 100) 		number of matrix rows
]])

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- create matrices
A = torch.rand(numRows,numRows)
B = torch.rand(numRows,numRows)

-- start timer
timer = torch.Timer()

-- multiply matrices
C = torch.mm(A,B)

-- show elapsed time
print('Time elapsed for multiplication of two matrices with ' .. numRows .. ' rows : ' .. timer:time().real .. ' seconds')

