#ifndef _CUDA_ARITHMETIC_
#define _CUDA_ARITHMETIC_

#include "cuda_tensor.h"

template<class T_a, class T_b, class T_c>
void tensor_element_wise_add(const CudaTensor<T_a>* a, const CudaTensor<T_b>* b, CudaTensor<T_c>* c);

template<class T_a, class T_b, class T_c>
void tensor_element_wise_add(const CudaTensor<T_a>* a, T_b b, CudaTensor<T_c>* c);

template<class T_a, class T_b, class T_c>
void tensor_element_wise_mul(const CudaTensor<T_a>* a, const CudaTensor<T_b>* b, CudaTensor<T_c>* c);

template<class T_a, class T_b, class T_c>
void tensor_element_wise_mul(const CudaTensor<T_a>* a, T_b b, CudaTensor<T_c>* c);

template<class T_a, class T_b, class T_c>
void tensor_element_wise_div(const CudaTensor<T_a>* a, const CudaTensor<T_b>* b, CudaTensor<T_c>* c);

#endif
