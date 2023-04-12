import numpy as np
import math
from numba import cuda, float64
from timeit import default_timer as timer

THREADSPERBLOCK = 16
MINIMUMBLOCKSIZE = 28
EPS = 1e-10

def config():
	#np.seterr(all="none")
	pass

def mse_cpu(predicted,target): 
	error = np.sum(np.square(predicted - target))/2.0
	return error

@cuda.jit
def mse(predicted, target, result):
    pos = cuda.grid(1)

    if pos < len(predicted):
        diff = predicted[pos] - target[pos]
        diff = (diff * diff) / 2.0
        cuda.atomic.add(result, 0, diff)

def mse_derivate_cpu(predicted,target):
	return (predicted - target)

@cuda.jit
def mse_derivate(predicted, target, result):
    pos = cuda.grid(1)

    if pos < len(predicted):
        result[0, pos] = predicted[0, pos] - target[0, pos]

def softmax_cpu(z):
	z = np.exp(z)
	sumT = z.sum()
	z = z/sumT
	return z

@cuda.jit
def softmax_p1(arr, res):
	x = cuda.grid(1)

	if x < arr.shape[1]:
		arr[0, x] = math.exp(arr[0, x])
		cuda.atomic.add(res, 0, arr[0, x])

@cuda.jit
def softmax_p2(arr, sumT):
	x = cuda.grid(1)

	if x < arr.shape[1]:
		arr[0, x] = arr[0, x] / sumT[0]

def softmax_derivate_cpu(z,alpha):
	soft = np.exp(z)
	S = soft.sum()
	beta = (alpha*soft).sum()/S
	soft = soft*(alpha - beta)/S
	return soft

@cuda.jit
def softmax_sum_derivate(arr, alpha, simple_sum, sum_times_alpha):
	x = cuda.grid(1)

	if x < arr.shape[1]:
		arr[0, x] = math.exp(arr[0, x])
		cuda.atomic.add(simple_sum, 0, arr[0, x])
		cuda.atomic.add(sum_times_alpha, 0, arr[0, x] * alpha[0, x])

@cuda.jit
def softmax_derivate(arr, alpha, simple_sum, sum_times_alpha):
	x = cuda.grid(1)
 
	if x < arr.shape[1]:
		arr[0, x] = (arr[0, x] * (alpha[0, x] - (sum_times_alpha[0] / simple_sum[0]))) / simple_sum[0]

def sigmoid2_cpu(z):
	return 2.0*(1.0/(1.0 + np.exp(-z))) - 1.0 # (-1,1)

@cuda.jit
def sigmoid2(arr):
	x = cuda.grid(1)

	if x < arr.shape[1]:
		arr[0, x] = 2.0 * (1.0 / (1.0 + math.exp(-arr[0, x]))) - 1.0

def sigmoid2_derivate_cpu(z,alpha):
    return alpha*(2.0*np.exp(-z)/((1.0 + np.exp(-z))*(1.0 + np.exp(-z))))

@cuda.jit
def sigmoid2_derivate(arr, alpha):
	x = cuda.grid(1)

	if x < arr.shape[1]:
		arr[0, x] = alpha[0, x] * (2.0 * math.exp(-arr[0, x]) / ( (1.0 + math.exp(-arr[0, x])) * (1.0 + math.exp(-arr[0, x]))))

def dotMatrix_cpu(x,w,b):
	return x.dot(w) + b

@cuda.jit
def dotMatrix(arr, A, B, C):
	sA = cuda.shared.array(shape=(THREADSPERBLOCK, THREADSPERBLOCK), dtype=float64)
	sB = cuda.shared.array(shape=(THREADSPERBLOCK, THREADSPERBLOCK), dtype=float64)
	
	x, y = cuda.grid(2)
	
	tx = cuda.threadIdx.x
	ty = cuda.threadIdx.y
	bpg = cuda.gridDim.y

	tmp = float64(0.)
	for i in range(bpg):
		sA[tx, ty] = float64(0.)
		sB[tx, ty] = float64(0.)

		if ty + i * THREADSPERBLOCK < A.shape[1] and x < A.shape[0]:
			sA[tx, ty] = A[x, ty + i * THREADSPERBLOCK]
		
		if tx + i * THREADSPERBLOCK < B.shape[0] and y < B.shape[1]:
			sB[tx, ty] = B[tx + i * THREADSPERBLOCK, y]
		
		cuda.syncthreads()
		
		for j in range(THREADSPERBLOCK):
			tmp += sA[tx, j] * sB[j, ty]
		
		cuda.syncthreads()

	if x < arr.shape[0] and y < arr.shape[1]:
		arr[x, y] = C[x, y] + tmp

def dotMatrix_derivate_cpu(x,w,alpha):
	return alpha.dot(w.transpose())

@cuda.jit
def dotMatrix_derivate(arr, w, alpha):
	sAlpha = cuda.shared.array(shape=(THREADSPERBLOCK, THREADSPERBLOCK), dtype=float64)
	sWTranspose = cuda.shared.array(shape=(THREADSPERBLOCK, THREADSPERBLOCK), dtype=float64)

	x, y = cuda.grid(2)
	
	tx = cuda.threadIdx.x
	ty = cuda.threadIdx.y
	bpg = cuda.gridDim.y

	tmp = float64(0.)
	for i in range(bpg):
		sAlpha[tx, ty] = float64(0.)
		sWTranspose[ty, tx] = float64(0.)

		if x < alpha.shape[0] and ty + i * THREADSPERBLOCK < alpha.shape[1]:
			sAlpha[tx, ty] = alpha[x, ty + i * THREADSPERBLOCK]
		
		if y < w.shape[0] and tx + i * THREADSPERBLOCK < w.shape[1]:
			sWTranspose[ty, tx] = w[y, tx + i * THREADSPERBLOCK]
		
		cuda.syncthreads()

		for j in range(THREADSPERBLOCK):
			tmp += sAlpha[tx, j] * sWTranspose[ty, j]
		
		cuda.syncthreads()

	if x < arr.shape[0] and y < arr.shape[1]:
		arr[x, y] = tmp

def transposeDot_cpu(x, derror):
	return x.transpose().dot(derror)

@cuda.jit
def transposeDot(arr, A, B):
	x, y = cuda.grid(2)

	if x < arr.shape[0] and y < arr.shape[1]:
		arr[x, y] = A[0, x] * B[0, y]

def updateWeights_cpu(w, eta, nabla):
	w = w - (eta * nabla)
	return w

@cuda.jit
def updateWeights(w, eta, nabla):
	x, y = cuda.grid(2)

	if x < w.shape[0] and y < w.shape[1]:
		w[x, y] -= eta[0] * nabla[x, y]

# unit tests
def mse_test():
    LEN_ARRAY = 5000

    stream = cuda.stream()

    predicted = np.random.randn(LEN_ARRAY)
    target = np.random.randn(LEN_ARRAY)
    result = np.zeros(1)

    d_predicted = cuda.to_device(predicted, stream=stream)
    d_target = cuda.to_device(target, stream=stream)
    d_result = cuda.to_device(result, stream=stream)

    cpu_answer = mse_cpu(predicted, target)

    BLOCKSPERGRID = max(MINIMUMBLOCKSIZE, (LEN_ARRAY + THREADSPERBLOCK - 1) // THREADSPERBLOCK)

    mse[BLOCKSPERGRID, THREADSPERBLOCK](d_predicted, d_target, d_result)

    h_result = d_result.copy_to_host(stream=stream)

    assert abs(h_result[0] - cpu_answer) <= EPS

def mse_derivate_test():
    LEN_ARRAY = 200000

    stream = cuda.stream()

    predicted = np.random.randn(1, LEN_ARRAY)
    target = np.random.randn(1, LEN_ARRAY)
    result = np.random.randn(1, LEN_ARRAY)

    d_predicted = cuda.to_device(predicted, stream=stream)
    d_target = cuda.to_device(target, stream=stream)
    d_result = cuda.to_device(result, stream=stream)

    r_cpu = mse_derivate_cpu(predicted[0], target[0])
    
    needblocksgrid = (LEN_ARRAY + THREADSPERBLOCK - 1) // THREADSPERBLOCK
    BLOCKSPERGRID = max(MINIMUMBLOCKSIZE, needblocksgrid)

    mse_derivate[BLOCKSPERGRID, THREADSPERBLOCK](d_predicted, d_target, d_result)

    r_gpu = d_result.copy_to_host(stream=stream)

    for i in range(len(r_cpu)):
        assert abs(r_cpu[i] - r_gpu[0, i]) <= EPS

def softmax_test():
	LEN_ARRAY = 2000000

	stream = cuda.stream()

	arr = np.random.randn(1, LEN_ARRAY)
	arr_gpu = cuda.to_device(arr, stream=stream)
	arr_cpu = np.copy(arr)

	arr_cpu = softmax_cpu(arr_cpu)

	res = np.zeros(1)
	res_gpu = cuda.to_device(res, stream=stream)

	threads = THREADSPERBLOCK
	blockspergrid = max(MINIMUMBLOCKSIZE, (LEN_ARRAY + THREADSPERBLOCK - 1) // THREADSPERBLOCK)

	softmax_p1[blockspergrid, threads](arr_gpu, res_gpu)
	softmax_p2[blockspergrid, threads](arr_gpu, res_gpu)

	out = arr_gpu.copy_to_host(stream=stream)

	for i in range(LEN_ARRAY):
		assert abs(out[0][i] - arr_cpu[0][i]) <= EPS

def softmax_derivate_test():
	LEN_ARRAY = 200000

	stream = cuda.stream()

	z = np.random.randn(1, LEN_ARRAY)
	alpha = np.random.randn(1, LEN_ARRAY)

	z_cpu = np.copy(z)
	alpha_cpu = np.copy(alpha)
	
	z_cpu = softmax_derivate_cpu(z_cpu, alpha_cpu)

	blockspergrid = (LEN_ARRAY + THREADSPERBLOCK - 1) // THREADSPERBLOCK

	z_gpu = cuda.to_device(z, stream=stream)
	alpha_gpu = cuda.to_device(alpha, stream=stream)

	simple_sum = np.zeros(1)
	sum_times_alpha = np.zeros(1)

	simple_sum_gpu = cuda.to_device(simple_sum, stream=stream)
	sum_times_alpha_gpu = cuda.to_device(sum_times_alpha, stream=stream)

	softmax_sum_derivate[blockspergrid, THREADSPERBLOCK](z_gpu, alpha_gpu, simple_sum_gpu, sum_times_alpha_gpu)
	softmax_derivate[blockspergrid, THREADSPERBLOCK](z_gpu, alpha_gpu, simple_sum_gpu, sum_times_alpha_gpu)

	ans_gpu = z_gpu.copy_to_host(stream=stream)

	for i in range(LEN_ARRAY):
		assert abs(ans_gpu[0, i] - z_cpu[0, i]) <= EPS

def sigmoid2_test():
	LEN_ARRAY = 2000000

	stream = cuda.stream()

	z = np.random.randn(1, LEN_ARRAY)

	z_cpu = np.copy(z)

	z_cpu = sigmoid2_cpu(z_cpu)

	blockspergrid = (LEN_ARRAY + THREADSPERBLOCK - 1) // THREADSPERBLOCK

	z_gpu = cuda.to_device(z, stream=stream)

	sigmoid2[blockspergrid, THREADSPERBLOCK](z_gpu)

	ans_gpu = z_gpu.copy_to_host(stream=stream)

	for i in range(LEN_ARRAY):
		assert abs(ans_gpu[0, i] - z_cpu[0, i]) <= EPS

def sigmoid2_derivate_test():
	LEN_ARRAY = 200000

	z = np.random.randn(1, LEN_ARRAY)
	alpha = np.random.randn(1, LEN_ARRAY)

	z_cpu = np.copy(z)
	alpha_cpu = np.copy(alpha)

	z_cpu = sigmoid2_derivate_cpu(z_cpu, alpha_cpu)

	blockspergrid = (LEN_ARRAY + THREADSPERBLOCK - 1) // (THREADSPERBLOCK)

	stream = cuda.stream()

	z_gpu = cuda.to_device(z, stream=stream)
	alpha_gpu = cuda.to_device(alpha, stream=stream)

	sigmoid2_derivate[blockspergrid, THREADSPERBLOCK](z_gpu, alpha_gpu)

	ans_gpu = z_gpu.copy_to_host(stream=stream)

	for i in range(LEN_ARRAY):
		assert abs(ans_gpu[0, i] - z_cpu[0, i]) <= EPS

def dotMatrix_test():
	LEN_ARRAY1 = 2000
	LEN_ARRAY2 = 3000

	x = np.random.randn(1, LEN_ARRAY1)
	w = np.random.randn(LEN_ARRAY1, LEN_ARRAY2)
	b = np.random.randn(1, LEN_ARRAY2)
	arr = np.zeros(b.shape)

	stream = cuda.stream()

	x_gpu = cuda.to_device(x, stream=stream)
	w_gpu = cuda.to_device(w, stream=stream)
	b_gpu = cuda.to_device(b, stream=stream)
	arr_gpu = cuda.to_device(arr, stream=stream)

	res = dotMatrix_cpu(x,w,b)

	grid_x_max = max(x_gpu.shape[0], w_gpu.shape[0])
	grid_y_max = max(x_gpu.shape[1], w_gpu.shape[1])
	blockspergrid_x = (grid_x_max + THREADSPERBLOCK - 1) // THREADSPERBLOCK
	blockspergrid_y = (grid_y_max + THREADSPERBLOCK - 1) // THREADSPERBLOCK
	blockspergrid = (blockspergrid_x, blockspergrid_y)

	dotMatrix[blockspergrid, (THREADSPERBLOCK, THREADSPERBLOCK)](arr_gpu, x_gpu, w_gpu, b_gpu)
	arr = arr_gpu.copy_to_host(stream=stream)

	for i in range(LEN_ARRAY2):
		assert abs(arr[0, i] - res[0, i]) <= EPS

def dotMatrix_derivate_test():
	LEN_ARRAY1 = 2000
	LEN_ARRAY2 = 3000

	x = np.random.randn(1, 1)
	w = np.random.randn(LEN_ARRAY1, LEN_ARRAY2)
	alpha = np.random.randn(1, LEN_ARRAY2)
	arr = np.zeros([1, LEN_ARRAY1])

	stream = cuda.stream()

	w_gpu = cuda.to_device(w, stream=stream)
	alpha_gpu = cuda.to_device(alpha, stream=stream)
	arr_gpu = cuda.to_device(arr, stream=stream)

	res = dotMatrix_derivate_cpu(x, w, alpha)

	grid_x_max = max(w_gpu.shape[0], alpha_gpu.shape[0])
	grid_y_max = max(w_gpu.shape[1], alpha_gpu.shape[1])
	blockspergrid_x = (grid_x_max + THREADSPERBLOCK - 1) // THREADSPERBLOCK
	blockspergrid_y = (grid_y_max + THREADSPERBLOCK - 1) // THREADSPERBLOCK
	blockspergrid = (blockspergrid_x, blockspergrid_y)

	dotMatrix_derivate[blockspergrid, (THREADSPERBLOCK, THREADSPERBLOCK)](arr_gpu, w_gpu, alpha_gpu)

	arr = arr_gpu.copy_to_host(stream=stream)

	for i in range(LEN_ARRAY1):
		assert abs(arr[0, i] - res[0, i]) <= EPS

def transposeDot_test():
	LEN_ARRAY1 = 1000
	LEN_ARRAY2 = 2000

	A = np.random.randn(1, LEN_ARRAY1)
	B = np.random.randn(1, LEN_ARRAY2)
	C = np.zeros([LEN_ARRAY1, LEN_ARRAY2])

	stream = cuda.stream()

	a_gpu = cuda.to_device(A, stream=stream)
	b_gpu = cuda.to_device(B, stream=stream)
	c_gpu = cuda.to_device(C, stream=stream)

	res = transposeDot_cpu(A, B)

	blockspergrid_x = (LEN_ARRAY1 + THREADSPERBLOCK - 1) // THREADSPERBLOCK
	blockspergrid_y = (LEN_ARRAY2 + THREADSPERBLOCK - 1) // THREADSPERBLOCK
	BLOCKSPERGRID = (blockspergrid_x, blockspergrid_y)
	transposeDot[BLOCKSPERGRID, (THREADSPERBLOCK, THREADSPERBLOCK)](c_gpu, a_gpu, b_gpu)

	c = c_gpu.copy_to_host(stream=stream)

	for i in range(LEN_ARRAY1):
		for j in range(LEN_ARRAY2):
			assert abs(c[i, j] - res[i, j]) <= EPS

def updateWeights_test():
	LEN_ARRAY1 = 1000
	LEN_ARRAY2 = 2000

	w = np.random.randn(LEN_ARRAY1, LEN_ARRAY2)
	nabla = np.random.randn(LEN_ARRAY1, LEN_ARRAY2)
	eta = np.random.randn(1)

	stream = cuda.stream()

	w_gpu = cuda.to_device(w, stream=stream)
	nabla_gpu = cuda.to_device(nabla, stream=stream)
	eta_gpu = cuda.to_device(eta, stream=stream)

	w = updateWeights_cpu(w, eta[0], nabla)

	blockspergrid_x = (LEN_ARRAY1 + THREADSPERBLOCK - 1) // THREADSPERBLOCK
	blockspergrid_y = (LEN_ARRAY2 + THREADSPERBLOCK - 1) // THREADSPERBLOCK
	BLOCKSPERGRID = (blockspergrid_x, blockspergrid_y)

	updateWeights[BLOCKSPERGRID, (THREADSPERBLOCK, THREADSPERBLOCK)](w_gpu, eta_gpu, nabla_gpu)

	w2 = w_gpu.copy_to_host(stream=stream)

	for i in range(LEN_ARRAY1):
		for j in range(LEN_ARRAY2):
			assert abs(w[i, j] - w2[i, j]) <= EPS

def test():
	tests = [softmax_test, softmax_derivate_test, sigmoid2_test, sigmoid2_derivate_test, dotMatrix_test,
	  		dotMatrix_derivate_test, transposeDot_test, updateWeights_test]

	print('Tests started')
	for currentTest in tests:
		ok = True
		try:
			currentTest()
		except:
			ok = False
		status = "{}".format('.' if ok else 'F')
		print(status, end='')

if __name__ == "__main__":
	test()