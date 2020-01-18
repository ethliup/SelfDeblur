#ifndef _CUDA_IMAGE_
#define _CUDA_IMAGE_

#include <cuda_runtime.h>
#include <stdio.h>

/*
 * Memory layout pattern: Channel->W->H->B->N
 */
template<class T>
class CudaTensor {
public:
    CudaTensor(){}

   	CudaTensor(int B, int C, int H, int W)
	: mN(1)
	, mB(B)
	, mC(C)
	, mH(H)
	, mW(W)
	, mNumElements(B*C*H*W) {
        cudaMalloc(&m_cuda_data_ptr, mNumElements * sizeof(T));
        m_owner = true;
    }

    CudaTensor(T* data_ptr, int B, int C, int H, int W)
            : mN(1)
            , mB(B)
            , mC(C)
            , mH(H)
            , mW(W)
            , mNumElements(B*C*H*W){
            m_owner = false;
            m_cuda_data_ptr=data_ptr;
    }

    void clone()
    {
        if(m_owner==false) {
            T *temp;
            cudaMalloc(&temp, this->mNumElements * sizeof(T));
            cudaMemcpy(temp, this->m_cuda_data_ptr, mNumElements * sizeof(T), cudaMemcpyDeviceToDevice);
            m_cuda_data_ptr = temp;
            m_owner = true;
        }
    }

    CudaTensor(int N, int B, int C, int H, int W)
            : mN(N)
            , mB(B)
            , mC(C)
            , mH(H)
            , mW(W)
            , mNumElements(N*B*C*H*W) {
        m_owner = true;
        cudaMalloc(&m_cuda_data_ptr, mNumElements * sizeof(T));
    }

	~CudaTensor() {
   	    if(m_owner) cudaFree(m_cuda_data_ptr);

        mN = 0;
        mB = 0;
        mC = 0;
        mH = 0;
        mW = 0;
        mNumElements = 0;
    }

	void reset(int value=0) {
        cudaMemset(m_cuda_data_ptr, value, mNumElements * sizeof(T));
    }

    T* operator[](unsigned int n)
    {
        T* ptr = (T*)((char*)m_cuda_data_ptr + n*mB*mC*mH*mW*sizeof(T));
        return ptr;
    }

    T* operator()(int B, int C, int H, int W, bool channels_first=false)
    {
        T* ptr=nullptr;
        if(channels_first) // BCHW
        {
            ptr = (T*)((char*)m_cuda_data_ptr +  B*mH*mW*mC*sizeof(T) + C*mH*mW*sizeof(T) + H*mW*sizeof(T) + W*sizeof(T));
        }
        else // BHWC
        {
            ptr = (T*)((char*)m_cuda_data_ptr +  B*mC*mH*mW*sizeof(T) + H*mC*mW*sizeof(T) + W*mC*sizeof(T) + C*sizeof(T));
        }
        return ptr;
    }

    T cpu_element(int B, int C, int H, int W, bool channels_first=false)
    {
        T* ptr=nullptr; 
        if(channels_first) // BCHW
        {
            ptr = (T*)((char*)m_cuda_data_ptr +  B*mH*mW*mC*sizeof(T) + C*mH*mW*sizeof(T) + H*mW*sizeof(T) + W*sizeof(T));  
        }
        else // BHWC
        {
            ptr = (T*)((char*)m_cuda_data_ptr +  B*mC*mH*mW*sizeof(T) + H*mC*mW*sizeof(T) + W*mC*sizeof(T) + C*sizeof(T));            
        }
        T value;
        cudaMemcpy(&value, ptr, sizeof(T), cudaMemcpyDeviceToHost);
        return value;
    }

    __host__ __device__ T* data_ptr()  const { return m_cuda_data_ptr;}
    __host__ __device__ T* data_ptr()        { return m_cuda_data_ptr;}
    __host__ __device__ int num_elem() const { return mNumElements; }

    __host__ __device__ int N() const { return mN; }
    __host__ __device__ int B() const { return mB; }
    __host__ __device__ int C() const { return mC; }
    __host__ __device__ int H() const { return mH; }
    __host__ __device__ int W() const { return mW; }


private:
	T* m_cuda_data_ptr;
    bool m_owner;
    int mN;
    int mB;
    int mC;
    int mH;
    int mW;
    int mNumElements;
};

#endif
