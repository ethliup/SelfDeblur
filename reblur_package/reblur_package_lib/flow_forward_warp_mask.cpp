#include "flow_forward_warp_mask.h"
#include "cuda_common.h"
#include "cuda_arithmetic.h"
#include "cuda_geometry.h"

Flow_forward_warp_mask::Flow_forward_warp_mask(at::Tensor grid) 
{
    mB = grid.size(0);
    mH = grid.size(2);
    mW = grid.size(3);

    int *temp;
    cudaMalloc(&temp, mB * 2 * mH * mW * sizeof(int));
    from_pytorch_mem_layout<int>(mB, 2, mH, mW, grid.data<int>(), temp);
    m_grid = new CudaTensor<int>(temp, mB, 2, mH, mW);
    m_grid->clone();
    cudaFree(temp);

    m_internal_mem_flow = new CudaTensor<float>(mB, 2, mH, mW);
    m_internal_mem_mask = new CudaTensor<float>(mB, 1, mH, mW);
}

Flow_forward_warp_mask::~Flow_forward_warp_mask() 
{
    delete m_grid;
    delete m_internal_mem_flow;
    delete m_internal_mem_mask;
}

void Flow_forward_warp_mask::get_mask(at::Tensor flow, at::Tensor mask) 
{
    from_pytorch_mem_layout<float>(mB, 2, mH, mW, flow.data<float>(), m_internal_mem_flow->data_ptr());

    // get warped mesh vertices positions
    tensor_element_wise_add<int, float, float>(m_grid, m_internal_mem_flow, m_internal_mem_flow);

    m_internal_mem_mask->reset(0);

    mask_via_flow_forward_warp(m_internal_mem_flow->data_ptr(), mB, mH, mW, m_internal_mem_mask->data_ptr());

    to_pytorch_mem_layout<float>(mB, 1, mH, mW, m_internal_mem_mask->data_ptr(), mask.data<float>());
}