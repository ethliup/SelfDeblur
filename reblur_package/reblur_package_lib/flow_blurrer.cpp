#include "flow_blurrer.h"
#include "cuda_common.h"
#include "cuda_arithmetic.h"
#include "cuda_geometry.h"

Flow_blurrer::Flow_blurrer(at::Tensor grid, at::Tensor mesh_faces, int nChannels_mesh_texture, int tar_image_H, int tar_image_W, int blur_kernel_size)
{
    int B = grid.size(0);
    int mesh_H = grid.size(2);
    int mesh_W = grid.size(3);
    
    int *temp;
    cudaMalloc(&temp, B * 2 * mesh_H * mesh_W * sizeof(int));
    from_pytorch_mem_layout(B, 2, mesh_H, mesh_W, grid.data<int>(), temp);
    m_grid = new CudaTensor<int>(temp, B, 2, mesh_H, mesh_W);
    m_grid->clone();
    cudaFree(temp);

    int nFacesPerBatch = mesh_faces.size(1);
    m_face = new CudaTensor<int>(mesh_faces.data<int>(), B, 1, nFacesPerBatch, 3);
    m_face->clone();

    m_nHimage = tar_image_H;
    m_nWimage = tar_image_W;
    m_nBlurKernelSize = blur_kernel_size;
    m_nFacesPerBatch = nFacesPerBatch;

    m_renderer = new Cuda_renderer_flow(nChannels_mesh_texture, tar_image_H, tar_image_W, nFacesPerBatch, B);

    m_internal_mem_texture = new CudaTensor<float>(B, nChannels_mesh_texture, mesh_H, mesh_W);
    m_internal_mem_flow = new CudaTensor<float>(B, 2, mesh_H, mesh_W);
    m_buffer_rearranged_texture = new CudaTensor<float>(B, nChannels_mesh_texture, nFacesPerBatch, 3);
    m_buffer_rearranged_vertex_xy = new CudaTensor<float>(blur_kernel_size/2 * 2 + 1, B, 2, nFacesPerBatch, 3);
    m_buffer_interpolated_flow = new CudaTensor<float>(B, 2, mesh_H, mesh_W);

    m_render_image = new CudaTensor<float>(blur_kernel_size/2 * 2 + 1, B, nChannels_mesh_texture, tar_image_H, tar_image_W);
    m_render_mask = new CudaTensor<float>(blur_kernel_size/2 * 2 + 1, B, 1, tar_image_H, tar_image_W);
    m_render_weight = new CudaTensor<float>(blur_kernel_size/2 * 2 + 1, B, 3, tar_image_H, tar_image_W);
    m_render_face_index = new CudaTensor<int>(blur_kernel_size/2 * 2 + 1, B, 1, tar_image_H, tar_image_W);

    m_internal_blur_image = new CudaTensor<float>(B, nChannels_mesh_texture, tar_image_H, tar_image_W);
    m_internal_mask_image = new CudaTensor<float>(B, 1, tar_image_H, tar_image_W);

    m_internal_grad_image = new CudaTensor<float>(B, nChannels_mesh_texture, tar_image_H, tar_image_W);
    m_grad_rearranged_vertex_xy = new CudaTensor<float>(B, 2, nFacesPerBatch, 3);
    m_grad_rearranged_texture = new CudaTensor<float>(B, nChannels_mesh_texture, nFacesPerBatch, 3);
    m_grad_vertex_xy = new CudaTensor<float>(B, 2, mesh_H, mesh_W);
    m_grad_texture = new CudaTensor<float>(B, nChannels_mesh_texture, mesh_H, mesh_W);
}

Flow_blurrer::~Flow_blurrer()
{
    delete m_grid;
    delete m_face;
    delete m_renderer;
    delete m_internal_mem_texture;
    delete m_internal_mem_flow;
    delete m_buffer_rearranged_texture;
    delete m_buffer_rearranged_vertex_xy;
    delete m_buffer_interpolated_flow;
    delete m_render_image;
    delete m_render_mask;
    delete m_render_weight;
    delete m_render_face_index;
    delete m_internal_blur_image;
    delete m_internal_mask_image;
    delete m_internal_grad_image;
    delete m_grad_rearranged_vertex_xy;
    delete m_grad_rearranged_texture;
    delete m_grad_vertex_xy;
    delete m_grad_texture;
}

void Flow_blurrer::forward(at::Tensor src_texture, at::Tensor flow_middle_to_last, at::Tensor tar_image, at::Tensor tar_mask)
{
    m_render_image->reset(0);
    m_render_mask->reset(0);
    m_render_weight->reset(0);
    m_render_face_index->reset(-1);
    m_internal_blur_image->reset(0);
    m_internal_mask_image->reset(0);
    tensor_element_wise_add<float, float, float>(m_internal_mask_image, 1.0, m_internal_mask_image);

    // convert pytorch mem layout to our mem layout
    from_pytorch_mem_layout(src_texture.size(0),
                            src_texture.size(1),
                            src_texture.size(2),
                            src_texture.size(3),
                            src_texture.data<float>(),
                            m_internal_mem_texture->data_ptr());
    from_pytorch_mem_layout(flow_middle_to_last.size(0),
                            2,
                            flow_middle_to_last.size(2),
                            flow_middle_to_last.size(3),
                            flow_middle_to_last.data<float>(),
                            m_internal_mem_flow->data_ptr());

    rearrange_vertices(m_internal_mem_texture->data_ptr(),
                       m_face->data_ptr(),
                       src_texture.size(2)*src_texture.size(3),
                       m_face->H(),
                       src_texture.size(0),
                       src_texture.size(1),
                       m_buffer_rearranged_texture->data_ptr());

    int blur_kernel_half_size = m_nBlurKernelSize / 2;
    for(int i = -blur_kernel_half_size; i <= blur_kernel_half_size; i++)
    {
        float fraction = ((float)i) / ((float)blur_kernel_half_size);
        tensor_element_wise_mul<float, float, float>(m_internal_mem_flow, fraction, m_buffer_interpolated_flow);
        tensor_element_wise_add<int, float, float>(m_grid, m_buffer_interpolated_flow, m_buffer_interpolated_flow);
        rearrange_vertices(m_buffer_interpolated_flow->data_ptr(), 
                           m_face->data_ptr(), 
                           src_texture.size(2)*src_texture.size(3),
                           m_face->H(), 
                           src_texture.size(0), 
                           2, 
                           (*m_buffer_rearranged_vertex_xy)[i + blur_kernel_half_size]);
        
        m_renderer->forward((*m_buffer_rearranged_vertex_xy)[i + blur_kernel_half_size],
                            m_buffer_rearranged_texture->data_ptr(),
                            (*m_render_image)[i + blur_kernel_half_size],
                            (*m_render_mask)[i + blur_kernel_half_size],
                            (*m_render_weight)[i + blur_kernel_half_size],
                            (*m_render_face_index)[i + blur_kernel_half_size]);
    }

    for(int i = 0; i < 2*blur_kernel_half_size+1; i++)
    {
        CudaTensor<float> latent_image_i = CudaTensor<float>((*m_render_image)[i], tar_image.size(0), tar_image.size(1), tar_image.size(2), tar_image.size(3));
        CudaTensor<float> latent_mask_i = CudaTensor<float>((*m_render_mask)[i], tar_image.size(0), 1, tar_image.size(2), tar_image.size(3));
        tensor_element_wise_add<float, float, float>(&latent_image_i, m_internal_blur_image, m_internal_blur_image);
        tensor_element_wise_mul<float, float, float>(&latent_mask_i, m_internal_mask_image, m_internal_mask_image);
    }
    tensor_element_wise_mul<float, float, float>(m_internal_blur_image, 1.0/(2 * blur_kernel_half_size+1), m_internal_blur_image);
    
    // convert our mem layout to pytorch layout
    to_pytorch_mem_layout<float>(tar_image.size(0),
                                 tar_image.size(1),
                                 tar_image.size(2),
                                 tar_image.size(3),
                                 m_internal_blur_image->data_ptr(),
                                 tar_image.data<float>());

    to_pytorch_mem_layout<float>(tar_image.size(0),
                                 1,
                                 tar_image.size(2),
                                 tar_image.size(3),
                                 m_internal_mask_image->data_ptr(),
                                 tar_mask.data<float>());
}

void Flow_blurrer::backward(at::Tensor grad_tar_image, at::Tensor grad_src_texture, at::Tensor grad_flow_middle_to_last)
{
    m_grad_rearranged_texture->reset(0);
    m_grad_rearranged_vertex_xy->reset(0);
    m_grad_vertex_xy->reset(0);
    m_grad_texture->reset(0);

    from_pytorch_mem_layout<float>(grad_tar_image.size(0),
                                   grad_tar_image.size(1),
                                   grad_tar_image.size(2),
                                   grad_tar_image.size(3),
                                   grad_tar_image.data<float>(),
                                   m_internal_grad_image->data_ptr());

    int blur_kernel_half_size = m_nBlurKernelSize / 2;
    tensor_element_wise_mul<float, float, float>(m_internal_grad_image, 1.0 / (blur_kernel_half_size * 2 + 1), m_internal_grad_image);

    for (int i = -blur_kernel_half_size; i <= blur_kernel_half_size; i++)
    {
        float fraction = ((float)i) / ((float)blur_kernel_half_size);
        m_grad_rearranged_vertex_xy->reset(0);
        m_grad_rearranged_texture->reset(0);
        
        m_renderer->backward((*m_buffer_rearranged_vertex_xy)[i + blur_kernel_half_size],
                             m_buffer_rearranged_texture->data_ptr(),
                             (*m_render_image)[i + blur_kernel_half_size],
                             (*m_render_face_index)[i + blur_kernel_half_size],
                             (*m_render_weight)[i + blur_kernel_half_size],
                             m_internal_grad_image->data_ptr(),
                             grad_tar_image.size(0),
                             m_nFacesPerBatch,
                             grad_tar_image.size(2),
                             grad_tar_image.size(3),
                             m_grad_rearranged_vertex_xy->data_ptr(),
                             m_grad_rearranged_texture->data_ptr());

        // the gradient is accumulating inside the function
        backward_rearrange_vertices(m_grad_rearranged_texture->data_ptr(), 
                                    m_face->data_ptr(),
                                    grad_src_texture.size(2) * grad_src_texture.size(3),
                                    m_nFacesPerBatch, 
                                    grad_src_texture.size(0), 
                                    grad_src_texture.size(1), 
                                    1.0, 
                                    m_grad_texture->data_ptr());

        backward_rearrange_vertices(m_grad_rearranged_vertex_xy->data_ptr(), 
                                    m_face->data_ptr(), 
                                    grad_flow_middle_to_last.size(2) * grad_flow_middle_to_last.size(3),
                                    m_nFacesPerBatch, 
                                    grad_flow_middle_to_last.size(0), 
                                    2, 
                                    fraction, 
                                    m_grad_vertex_xy->data_ptr());
    }

    to_pytorch_mem_layout<float>(grad_src_texture.size(0),
                                 grad_src_texture.size(1),
                                 grad_src_texture.size(2),
                                 grad_src_texture.size(3),
                                 m_grad_texture->data_ptr(),
                                 grad_src_texture.data<float>());

    to_pytorch_mem_layout<float>(grad_flow_middle_to_last.size(0),
                                 grad_flow_middle_to_last.size(1),
                                 grad_flow_middle_to_last.size(2),
                                 grad_flow_middle_to_last.size(3),
                                 m_grad_vertex_xy->data_ptr(),
                                 grad_flow_middle_to_last.data<float>());
}