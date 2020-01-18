#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

enum tex_type {TEX_U1, TEX_U2, TEX_U4, TEX_I2, TEX_I4, TEX_F1, TEX_F2, TEX_F4};

template<class T>
void cudaTextureToCudaMem(const cudaArray* input, T* output, tex_type type, int nOutputChans, int H, int W);

template<class T>
void from_pytorch_mem_layout(int B, int C, int H, int W, const T* src_data_ptr, T* dst_data_ptr);

template<class T>
void to_pytorch_mem_layout(int B, int C, int H, int W, const T* src_data_ptr, T* dst_data_ptr);

#endif