#ifndef _CORR_CUDA_CPP_
#define _CORR_CUDA_CPP_

#include <torch/torch.h>

class corr_cuda
{
public:
    corr_cuda();

    void forward(at::Tensor input1,
                at::Tensor input2,
                at::Tensor rbot1,
                at::Tensor rbot2,
                at::Tensor output,
                int pad_size,
                int kernel_size,
                int max_displacement,
                int stride1,
                int stride2,
                int corr_type_multiply,
                int batchSize,
                int nInputPlane,
                int nInputRows,
                int nInputCols);

    void backward(at::Tensor input1,
                at::Tensor input2,
                at::Tensor rbot1,
                at::Tensor rbot2,
                at::Tensor grad_output,
                at::Tensor grad_input1,
                at::Tensor grad_input2,
                int pad_size,
                int kernel_size,
                int max_displacement,
                int stride1,
                int stride2,
                int corr_type_multiply,
                int batchSize,
                int nInputPlane,
                int nInputRows,
                int nInputCols);
};

#endif
