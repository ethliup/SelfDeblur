#ifndef _FLOW_BLURRER_
#define _FLOW_BLURRER_

#include <torch/torch.h>
#include "cuda_tensor.h"
#include "cuda_renderer_flow.h"

class Flow_blurrer
{
public:
    Flow_blurrer(at::Tensor grid, at::Tensor mesh_faces, int nChannels_mesh_texture, int tar_image_H, int tar_image_W, int blur_kernel_size);
    ~Flow_blurrer();

    void forward(at::Tensor src_texture, at::Tensor flow_middle_to_last, at::Tensor tar_image, at::Tensor tar_mask);
    void backward(at::Tensor grad_tar_image, at::Tensor grad_src_texture, at::Tensor grad_flow_middle_to_last);

private:
    Cuda_renderer_flow* m_renderer;
    CudaTensor<int>* m_grid;
    CudaTensor<int>* m_face;

    CudaTensor<float>* m_internal_mem_texture;
    CudaTensor<float>* m_internal_mem_flow;
    CudaTensor<float>* m_buffer_rearranged_texture;
    CudaTensor<float>* m_buffer_rearranged_vertex_xy;
    CudaTensor<float>* m_buffer_interpolated_flow;

    CudaTensor<float>* m_render_image;
    CudaTensor<float>* m_render_mask;
    CudaTensor<float>* m_render_weight;
    CudaTensor<int>*   m_render_face_index;

    CudaTensor<float>* m_internal_blur_image;
    CudaTensor<float>* m_internal_mask_image;

    CudaTensor<float>* m_internal_grad_image;
    CudaTensor<float>* m_grad_rearranged_vertex_xy;
    CudaTensor<float>* m_grad_rearranged_texture;
    CudaTensor<float>* m_grad_vertex_xy;
    CudaTensor<float>* m_grad_texture;

    int m_nHimage;
    int m_nWimage;
    int m_nFacesPerBatch;
    int m_nBlurKernelSize;
};

#endif