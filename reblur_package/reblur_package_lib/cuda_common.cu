#include "cuda_common.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> texRef_u1;
texture<float,  cudaTextureType2D, cudaReadModeElementType> texRef_f1;
texture<int2,    cudaTextureType2D, cudaReadModeElementType>texRef_i2;
texture<int4,    cudaTextureType2D, cudaReadModeElementType>texRef_i4;
texture<uchar2, cudaTextureType2D, cudaReadModeElementType> texRef_u2;
texture<float2, cudaTextureType2D, cudaReadModeElementType> texRef_f2;
texture<float4, cudaTextureType2D, cudaReadModeElementType> texRef_f4;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef_u4;

inline int getNumTiles(int totalSize, int tileSize)
{
    const int div = totalSize / tileSize;
    return totalSize % tileSize == 0 ? div : div + 1;
}

template<class T>
__global__ void kernel_texture_to_memory(T* output, tex_type type, int nOutputChans, int H, int W) {
    const int X = blockIdx.x * blockDim.x + threadIdx.x;
    const int Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (!(X < W && Y < H)) return;

    int index = (Y * W + X) * nOutputChans;

    switch (type) {
        case TEX_U1: {
            output[index] = tex2D(texRef_u1, X, Y);
            break;
        }
        case TEX_U2: {
            uchar2 data = tex2D(texRef_u2, X, Y);
            const unsigned char data_array[2] = {data.x, data.y};
            for (int i = 0; i < nOutputChans; i++) {
                output[index + i] = data_array[i];
            }
            break;
        }
        case TEX_U4: {
            uchar4 data = tex2D(texRef_u4, X, Y);
            const unsigned char data_array[4] = {data.x, data.y, data.z, data.w};
            for (int i = 0; i < nOutputChans; i++) {
                output[index + i] = data_array[i];
            }
            break;
        }
        case TEX_I2: {
            int2 data = tex2D(texRef_i2, X, Y);
            const int data_array[2] = {data.x, data.y};
            for (int i = 0; i < nOutputChans; i++) {
                output[index + i] = data_array[i];
            }
            break;
        }
        case TEX_I4: {
            int4 data = tex2D(texRef_i4, X, Y);
            const int data_array[4] = {data.x, data.y, data.z, data.w};
            for (int i = 0; i < nOutputChans; i++) {
                output[index + i] = data_array[i];
            }
            break;
        }
        case TEX_F1: {
            float data = tex2D(texRef_f1, X, Y);
            output[index] = data;
            break;
        }
        case TEX_F2: {
            float2 data = tex2D(texRef_f2, X, Y);
            float data_array[2] = {data.x, data.y};
            for (int i = 0; i < nOutputChans; i++) {
                output[index + i] = data_array[i];
            }
            break;
        }
        case TEX_F4: {
            float4 data = tex2D(texRef_f4, X, Y);
            float data_array[4] = {data.x, data.y, data.z, data.w};
            for (int i = 0; i < nOutputChans; i++) {
                output[index + i] = data_array[i];
            }
            break;
        }
    }
}

template<class T>
__global__ void from_pytorch_mem_layout_kernel(int B, int C, int H, int W, const T* src_data_ptr, T* dst_data_ptr)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= B*C*H*W) return;

    int bn = i / (C*H*W); int remaining = i % (C*H*W);
    int ch = remaining / (H*W); remaining = remaining % (H*W);
    int r = remaining / W; remaining = remaining % W;
    int c = remaining;
    dst_data_ptr[bn * (H*W*C) + r * W * C + c * C + ch] = src_data_ptr[i];
}

template<class T>
__global__ void to_pytorch_mem_layout_kernel(int B, int C, int H, int W, const T* src_data_ptr, T* dst_data_ptr)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= B*C*H*W) return;
    int bn = i / (C*H*W); int remaining = i % (C*H*W);
    int r = remaining / (W*C); remaining = remaining % (W*C);
    int c = remaining / C;
    int ch = remaining % C;
    dst_data_ptr[bn*H*W*C + ch*H*W + r * W + c] = src_data_ptr[i];
}


template<class T>
void cudaTextureToCudaMem(const cudaArray* input, T* output, tex_type type, int nOutputChans, int H, int W)
{
    switch(type) {
        case TEX_U1:
            cudaBindTextureToArray(texRef_u1, input);
            break;
        case TEX_U2:
            cudaBindTextureToArray(texRef_u2, input);
            break;
        case TEX_U4:
            cudaBindTextureToArray(texRef_u4, input);
            break;
        case TEX_I2:
            cudaBindTextureToArray(texRef_i2, input);
            break;
        case TEX_I4:
            cudaBindTextureToArray(texRef_i4, input);
            break;
        case TEX_F1:
            cudaBindTextureToArray(texRef_f1, input);
            break;
        case TEX_F2:
            cudaBindTextureToArray(texRef_f2, input);
            break;
        case TEX_F4:
            cudaBindTextureToArray(texRef_f4, input);
            break;
    }

    dim3 gridDim = dim3(getNumTiles(W, 32), getNumTiles(H, 32));
    dim3 blockDim = dim3(32, 32);
    kernel_texture_to_memory<T><<<gridDim, blockDim>>>(output, type, nOutputChans, H, W);
    cudaDeviceSynchronize();

    switch(type) {
        case TEX_U1:
            cudaUnbindTexture(texRef_u1);
            break;
        case TEX_U2:
            cudaUnbindTexture(texRef_u2);
            break;
        case TEX_U4:
            cudaUnbindTexture(texRef_u4);
            break;
        case TEX_I2:
            cudaUnbindTexture(texRef_i2);
            break;
        case TEX_I4:
            cudaUnbindTexture(texRef_i4);
            break;
        case TEX_F1:
            cudaUnbindTexture(texRef_f1);
            break;
        case TEX_F2:
            cudaUnbindTexture(texRef_f2);
            break;
        case TEX_F4:
            cudaUnbindTexture(texRef_f4);
            break;
    }
}


template<class T>
void from_pytorch_mem_layout(int B, int C, int H, int W, const T* src_data_ptr, T* dst_data_ptr)
{
    const int threads = 512;
    const dim3 blocks ((B*C*H*W) / threads + 1);
    //printf("kernel BCHW %d %d %d %d\n", B, C, H, W);
    from_pytorch_mem_layout_kernel<<<blocks, threads>>>(B, C, H, W, src_data_ptr, dst_data_ptr);
    cudaDeviceSynchronize();
}

template<class T>
void to_pytorch_mem_layout(int B, int C, int H, int W, const T* src_data_ptr, T* dst_data_ptr)
{
    const int threads = 512;
    const dim3 blocks ((B*C*H*W) / threads + 1);
    to_pytorch_mem_layout_kernel<<<blocks, threads>>>(B, C, H, W, src_data_ptr, dst_data_ptr);
    cudaDeviceSynchronize();
}

/** Instatiate template functions **/
template void cudaTextureToCudaMem<unsigned char>(const cudaArray* input, unsigned char* output, tex_type type, int nOutputChans, int H, int W);
template void cudaTextureToCudaMem<int>(const cudaArray* input, int* output, tex_type type, int nOutputChans, int H, int W);
template void cudaTextureToCudaMem<float>(const cudaArray* input, float* output, tex_type type, int nOutputChans, int H, int W);

template void from_pytorch_mem_layout<int>(int B, int C, int H, int W, const int* src_data_ptr, int* dst_data_ptr);
template void from_pytorch_mem_layout<float>(int B, int C, int H, int W, const float* src_data_ptr, float* dst_data_ptr);
template void to_pytorch_mem_layout<float>(int B, int C, int H, int W, const float* src_data_ptr, float* dst_data_ptr);
template void to_pytorch_mem_layout<unsigned char>(int B, int C, int H, int W, const unsigned char* src_data_ptr, unsigned char* dst_data_ptr);