#include "cuda_arithmetic.h"

// + 
template<class T_a, class T_b, class T_c>
__global__ void kernel_tensor_element_wise_add(const T_a* a, const T_b* b, T_c* c, int nMaxElements)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nMaxElements) return;
    c[i] = a[i] + b[i];
}

template<class T_a, class T_b, class T_c>
__global__ void kernel_tensor_element_wise_add(const T_a* a, T_b b, T_c* c, int nMaxElements)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nMaxElements) return;
    c[i] = a[i] + b;
}

// mul
template<class T_a, class T_b, class T_c>
__global__ void kernel_tensor_element_wise_mul(const T_a* a, T_b b, T_c* c, int nMaxElements)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nMaxElements) return;
    c[i] = a[i] * b;
}

template<class T_a, class T_b, class T_c>
__global__ void kernel_tensor_element_wise_mul(const T_a* a, const T_b* b, T_c* c, int nMaxElements)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nMaxElements) return;
    c[i] = a[i] * b[i];
}

// div
template<class T_a, class T_b, class T_c>
__global__ void kernel_tensor_element_wise_div(const T_a* a, const T_b* b, T_c* c, int nMaxElements)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nMaxElements) return;
    c[i] = a[i] / (b[i]+1e-8);
}

/*********************************************/
/*              C++ wrappers                 */
/*********************************************/
template<class T_a, class T_b, class T_c>
void tensor_element_wise_add(const CudaTensor<T_a>* a, const CudaTensor<T_b>* b, CudaTensor<T_c>* c)
{
    int nThreads = 512;
    int nBlocks = a->num_elem() / nThreads + 1;
    const dim3 blocks (nBlocks);
    kernel_tensor_element_wise_add<T_a, T_b, T_c> << <blocks, nThreads>> > (a->data_ptr(), b->data_ptr(), c->data_ptr(), a->num_elem());
    cudaDeviceSynchronize();
}

template<class T_a, class T_b, class T_c>
void tensor_element_wise_add(const CudaTensor<T_a>* a, T_b b, CudaTensor<T_c>* c)
{
    int nThreads = 512;
    int nBlocks = a->num_elem() / nThreads + 1;
    const dim3 blocks (nBlocks);
    kernel_tensor_element_wise_add<T_a, T_b, T_c> << <blocks, nThreads>> > (a->data_ptr(), b, c->data_ptr(), a->num_elem());
    cudaDeviceSynchronize();
}

template<class T_a, class T_b, class T_c>
void tensor_element_wise_mul(const CudaTensor<T_a>* a, T_b b, CudaTensor<T_c>* c)
{
    const int nThreads = 512;
    const dim3 blocks (a->num_elem() / nThreads + 1);
    kernel_tensor_element_wise_mul<T_a, T_b, T_c> << <blocks, nThreads>> > (a->data_ptr(), b, c->data_ptr(), a->num_elem());
    cudaDeviceSynchronize();
}

template<class T_a, class T_b, class T_c>
void tensor_element_wise_mul(const CudaTensor<T_a>* a, const CudaTensor<T_b>* b, CudaTensor<T_c>* c)
{
    const int nThreads = 512;
    const dim3 blocks (a->num_elem() / nThreads + 1);
    kernel_tensor_element_wise_mul<T_a, T_b, T_c> << <blocks, nThreads>> > (a->data_ptr(), b->data_ptr(), c->data_ptr(), a->num_elem());
    cudaDeviceSynchronize();
}

template<class T_a, class T_b, class T_c>
void tensor_element_wise_div(const CudaTensor<T_a>* a, const CudaTensor<T_b>* b, CudaTensor<T_c>* c)
{
    const int nThreads = 512;
    const dim3 blocks (a->num_elem() / nThreads + 1);
    kernel_tensor_element_wise_div<T_a, T_b, T_c> << <blocks, nThreads>> > (a->data_ptr(), b->data_ptr(), c->data_ptr(), a->num_elem());
    cudaDeviceSynchronize();
}

template void tensor_element_wise_add<float, float, float>(const CudaTensor<float>* a, const CudaTensor<float>* b, CudaTensor<float>* c);
template void tensor_element_wise_add<int,   float, float>(const CudaTensor<int>*   a, const CudaTensor<float>* b, CudaTensor<float>* c);
template void tensor_element_wise_add<float, float, float>(const CudaTensor<float>* a, float b, CudaTensor<float>* c);
template void tensor_element_wise_add<int,   int,   int>  (const CudaTensor<int>*   a, int   b, CudaTensor<int>* c);
template void tensor_element_wise_mul<float, float, float>(const CudaTensor<float>* a, float b, CudaTensor<float>* c);
template void tensor_element_wise_mul<float, float, float>(const CudaTensor<float>* a, const CudaTensor<float>* b, CudaTensor<float>* c);
template void tensor_element_wise_div<float, float, float>(const CudaTensor<float>* a, const CudaTensor<float>* b, CudaTensor<float>* c);

