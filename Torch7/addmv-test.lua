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
   -N,--numRows      (default 10000) 	     number of matrix rows
   -i,--iterations   (default 10)         number of iterations
]])

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

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats
torch.setdefaulttensortype('torch.FloatTensor')

-- Matrices in the GPU
N = opt.numRows
A = torch.CudaTensor( N , N )
v = torch.CudaTensor( N )
w = torch.CudaTensor( N )
zeros = torch.CudaTensor( N )


-- fill matrices in cpu
A_cpu = torch.rand( N , N )
v_cpu = torch.rand( N )
zeros_cpu = torch.zeros(N)

-- tensors for the results
w_cpu = torch.zeros(N)
w[{}] = w_cpu

-- copy to the gpu
A[{}] = A_cpu
v[{}] = v_cpu
zeros[{}] = zeros_cpu


-- OpenBLAS test
-- start timer
timer = torch.Timer()
print('<torch> timer started')

-- perform matrix operation

for j = 1 , opt.iterations , 1 do
   w_cpu:zero()
   w_cpu:addmv( A_cpu , v_cpu )
end

-- show elapsed time
print('Time elapsed for ' , opt.iterations, ' matrix-vector addition with ' .. opt.numRows .. ' rows : ' .. timer:time().real .. ' seconds with OpenBLAS')

-- cuBLAS test
-- start timer
timer = torch.Timer()
print('<torch> timer started')

-- perform matrix operation

for j = 1 , opt.iterations , 1 do
   w:zero()
   w:addmv( A , v ) 
end

-- show elapsed time
print('Time elapsed for ' , opt.iterations, ' matrix-vector addition with ' .. opt.numRows .. ' rows : ' .. timer:time().real .. ' seconds with cuBLAS')


-- Compare result
local tester
tester = torch.Tester()
local tolerance = 0.01
local mistake = 0
for j = 1 , N , 1 do
   if math.abs(w_cpu[j] - w[j]) > tolerance then
      mistake = 1
   end
end
if mistake == 1 then
   print('<test> DIVERGENT result between cuBLAS and OpenBLAS')
   print("v : ", v , "v_cpu : " , v_cpu , "w : " , w , "w_cpu : " ,  w_cpu , " zeros : " , zeros , " zeros_cpu : " , zeros_cpu , " A : " , A , " A_cpu : " , A_cpu )
end
