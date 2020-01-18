#ifndef _FLOW_FORWARD_WARP_MASK_
#define _FLOW_FORWARD_WARP_MASK_

#include <torch/torch.h>
#include "cuda_tensor.h"

class Flow_forward_warp_mask
{
public:
    Flow_forward_warp_mask(at::Tensor grid);
    ~Flow_forward_warp_mask();

    void get_mask(at::Tensor flow, at::Tensor mask);

private:
    int mB, mH, mW;
    CudaTensor<int>* m_grid;
    CudaTensor<float>* m_internal_mem_flow;
    CudaTensor<float>* m_internal_mem_mask;
};

#endif