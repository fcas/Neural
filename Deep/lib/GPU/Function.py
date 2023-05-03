import math
from numba import cuda, float64

@cuda.jit
def copy(arr, A):
	x = cuda.grid(1)

	if x < arr.shape[1]:
		arr[0, x] = A[0, x]

@cuda.jit
def memset(arr):
	x = cuda.grid(1)

	if x < arr.shape[0]:
		arr[x] = float64(0.)

@cuda.jit
def memset2(arr):
	x, y = cuda.grid(2)

	if x < arr.shape[0] and y < arr.shape[1]:
		arr[x, y] = float64(0.)

@cuda.jit
def mse(result, predicted, target):
    pos = cuda.grid(1)

    if pos < predicted.shape[0]:
        diff = predicted[pos] - target[pos]
        diff = (diff * diff) / 2.0
        cuda.atomic.add(result, 0, diff)

@cuda.jit
def mse_derivate(result, predicted, target):
    pos = cuda.grid(1)

    if pos < predicted.shape[1]:
        result[0, pos] = predicted[0, pos] - target[0, pos]

@cuda.jit
def softmax_p1(arr, z, res):
	x = cuda.grid(1)

	if x < arr.shape[1]:
		arr[0, x] = math.exp(z[0, x])
		cuda.atomic.add(res, 0, arr[0, x])

@cuda.jit
def softmax_p2(arr, sumT):
	x = cuda.grid(1)

	if x < arr.shape[1]:
		arr[0, x] = arr[0, x] / sumT[0]

@cuda.jit
def softmax_sum_derivate(arr, z, alpha, simple_sum, sum_times_alpha):
	x = cuda.grid(1)

	if x < arr.shape[1]:
		value = z[0, x]
		value = math.exp(value)
		arr[0, x] = value
		cuda.atomic.add(simple_sum, 0, value)
		cuda.atomic.add(sum_times_alpha, 0, value * alpha[0, x])

@cuda.jit
def softmax_derivate(arr, alpha, simple_sum, sum_times_alpha):
	x = cuda.grid(1)
 
	if x < arr.shape[1]:
		value = arr[0, x]
		arr[0, x] = (value * (alpha[0, x] - (sum_times_alpha[0] / simple_sum[0]))) / simple_sum[0]

@cuda.jit
def sigmoid2(arr, A):
	x = cuda.grid(1)

	if x < arr.shape[1] and 0 < arr.shape[0]:
		arr[0, x] = 2.0 * (1.0 / (1.0 + math.exp(-A[0, x]))) - 1.0

@cuda.jit
def sigmoid2_derivate(arr, A, alpha):
	x = cuda.grid(1)

	if x < A.shape[1]:
		value = A[0, x]
		arr[0, x] = alpha[0, x] * (2.0 * math.exp(-value) / ( (1.0 + math.exp(-value)) * (1.0 + math.exp(-value))))

@cuda.jit
def sum(arr, C):
	x, y = cuda.grid(2)

	if x >= arr.shape[0] or y >= arr.shape[1]:
		return

	arr[x, y] += C[x, y]

@cuda.jit
def dotMatrix(arr, A, B):
	x, y, z = cuda.grid(3)

	if x >= A.shape[0] or y >= A.shape[1] or z >= B.shape[1]:
		return
	
	cuda.atomic.add(arr, (x, z), A[x, y] * B[y, z])

@cuda.jit
def dotMatrix_derivate(arr, w, alpha):
	x, y, z = cuda.grid(3)

	if x >= arr.shape[0] or y >= arr.shape[1] or z >= alpha.shape[1]:
		return

	cuda.atomic.add(arr, (x, y), alpha[x, z] * w[y, z])

@cuda.jit
def transposeDot(arr, A, B):
	x, y = cuda.grid(2)

	if x < arr.shape[0] and y < arr.shape[1]:
		arr[x, y] = A[0, x] * B[0, y]

@cuda.jit
def updateWeights(w, eta, nabla):
	x, y = cuda.grid(2)

	if x < w.shape[0] and y < w.shape[1]:
		w[x, y] -= eta[0] * nabla[x, y]