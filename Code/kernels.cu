template<typename T, typename U>
__global__ void addCUDA(T *a, U *b, T *c){
  *c = *a + *b;
}
template<typename T, typename U>
__global__ void subCUDA(T *a, U *b, T *c){
  *c = *a - *b;
}
template<typename T, typename U>
__global__ void mulCUDA(T *a, U *b, T *c){
  *c = *a * *b;
}
template<typename T, typename U>
__global__ void divCUDA(T *a, U *b, T *c){
  *c = *a / *b;
}

//array op array
template<typename T, typename U>
__global__ void addVecCUDA(T *a, U *b, T *c){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  c[index] = a[index] + b[index];
}

template<typename T, typename U>
__global__ void subVecCUDA(T *a, U *b, T *c){
    /* insert code to calculate the index properly using blockIdx.x, blockDim.x, threadIdx.x */
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  c[index] = a[index] - b[index];
}
template<typename T, typename U>
__global__ void mulVecCUDA(T *a, U *b, T *c){
  /* insert code to calculate the index properly using blockIdx.x, blockDim.x, threadIdx.x */
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  c[index] = a[index] * b[index];
}
template<typename T, typename U>
__global__ void divVecCUDA(T *a, U *b, T *c){
  /* insert code to calculate the index properly using blockIdx.x, blockDim.x, threadIdx.x */
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  c[index] = a[index] / b[index];
}
template<typename T, typename U>
__global__ void accVecCUDA(T *a, U *b, T *c, int count){
    /* insert code to calculate the index properly using blockIdx.x, blockDim.x, threadIdx.x */
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < count; ++i){
    c[index] = c[index] + a[index] + b[index];
  }
}

//pi
template<typename T>
__global__ void pi(T *result, int n) {
    T sum = T(0);
    for (int i = 0; i < n; i+=2){
      sum = sum + (T(1) / (T(2) * T(i) + T(1)));
      sum = sum - (T(1) / (T(2) * T(i+1) + T(1)));
    }
    *result = T(4)*sum;
}