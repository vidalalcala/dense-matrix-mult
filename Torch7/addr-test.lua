----------------------------------------------------------------------
-- A test for the OpenBLAS library with Torch7 tensors
--
-- (Jose V. Alcala-Burgos, 2014)
--

require 'torch'
require 'pl'
require 'cutorch'
print(  cutorch.getDeviceProperties(cutorch.getDevice()) )

-- Parse command-line options
local opt = lapp([[
   -t,--threads      (default 7)             number of threads
   -N,--numRows      (default 10002) 	     number of matrix rows
   -i,--iterations   (default 10)         number of iterations
]])

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats
torch.setdefaulttensortype('torch.FloatTensor')

-- Matrices in the GPU
N = opt.numRows
R = torch.CudaTensor( N , N )
v = torch.CudaTensor( N )
w = torch.CudaTensor( N )


-- fill matrices in cpu
v_cpu = torch.rand( N )
w_cpu = torch.rand( N )

-- tensors for the results
R_cpu = torch.zeros( N , N )

-- copy to the gpu
v[{}] = v_cpu
w[{}] = w_cpu
R[{}] = R_cpu

-- OpenBLAS test
-- start timer
timer = torch.Timer()
print('<torch> timer started')

-- perform matrix operation

for j = 1 , opt.iterations , 1 do
     R_cpu:addr( v_cpu , w_cpu )
end

-- show elapsed time
print('Time elapsed for ' , opt.iterations, ' additions of vector-vector product with ' .. opt.numRows .. ' rows : ' .. timer:time().real .. ' seconds with OpenBLAS')

-- cuBLAS test
-- start timer
timer = torch.Timer()
print('<torch> timer started')

-- perform matrix operation

for j = 1 , opt.iterations , 1 do
   R:addr( v , w ) 
end

-- show elapsed time
print('Time elapsed for ' , opt.iterations, ' aditions of vector-vector product with ' .. opt.numRows .. ' rows : ' .. timer:time().real .. ' seconds with cuBLAS')


-- Compare result
local tester
tester = torch.Tester()
local tolerance = 0.01
local mistake = 0
for j = 1 , N , 1 do
   if math.abs(R_cpu[1][j] - R[1][j]) > tolerance then
      mistake = 1
   end
end
if mistake == 1 then
   print('<test> DIVERGENT result between cuBLAS and OpenBLAS')
   print("v : ", v , "v_cpu : " , v_cpu , "w : " , w , "w_cpu : " ,  w_cpu , " R : " , R , " R_cpu : " , R_cpu )
end
