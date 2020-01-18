#include "cuda_renderer_flow.h"
#include "cuda_renderer.h"
#include <vector>
#include <cuda_runtime.h>

Cuda_renderer_flow::Cuda_renderer_flow(int nChannelsTexture, int H, int W, int nTrianglesPerBatch, int nBatches)
: m_image_H(H)
, m_image_W(W)
, m_nTrianglesPerBatch(nTrianglesPerBatch)
, m_nBatches(nBatches)
, m_nMaxTrianglesperPixel(20)
, m_nChannelTexture(nChannelsTexture)
{
    m_buffer_baycentric_coeffs = new CudaTensor<float>(nBatches, 1, nTrianglesPerBatch, 9);
    m_buffer_map_pixel_to_triangles = new CudaTensor<int>(nBatches, m_nMaxTrianglesperPixel, H, W);
}

Cuda_renderer_flow::~Cuda_renderer_flow()
{
    delete m_buffer_map_pixel_to_triangles;
    delete m_buffer_baycentric_coeffs;
}

void Cuda_renderer_flow::forward(const float* rearranged_mesh_vertex_xy,
                                 const float* rearranged_mesh_vertex_texture,
                                 float* rendered_image,
                                 float* rendered_mask,
                                 float* rendered_weight,
                                 int*   rendered_face_index)
{
    m_buffer_map_pixel_to_triangles->reset(-1);
    m_buffer_baycentric_coeffs->reset(0);

    forward_flow_renderer(rearranged_mesh_vertex_xy,
                          rearranged_mesh_vertex_texture,
                          m_nBatches,
                          m_nTrianglesPerBatch,
                          m_image_H,
                          m_image_W,
                          m_nMaxTrianglesperPixel,
                          m_nChannelTexture,
                          m_buffer_baycentric_coeffs->data_ptr(),
                          m_buffer_map_pixel_to_triangles->data_ptr(),
                          rendered_image,
                          rendered_weight,
                          rendered_face_index,
                          rendered_mask);
}

void Cuda_renderer_flow::backward(const float* rearranged_mesh_vertex_xy,
                                  const float* rearranged_mesh_vertex_texture,
                                  const float* rendered_image,
                                  const int*   rendered_face_index_map,
                                  const float* rendered_weight_map,
                                  const float* grad_rendered_image,
                                  const int    batch_size,
                                  const int    n_triangles_per_batch,
                                  const int    image_size_H,
                                  const int    image_size_W,
                                  float*  grad_rearranged_mesh_vertex_xy,
                                  float*  grad_rearranged_mesh_vertex_texture) 
{
    backward_flow_renderer(rearranged_mesh_vertex_xy, 
                           rearranged_mesh_vertex_texture, 
                           rendered_image, 
                           rendered_face_index_map, 
                           rendered_weight_map, 
                           grad_rendered_image, 
                           m_nChannelTexture, 
                           batch_size,
                           n_triangles_per_batch, 
                           image_size_H, 
                           image_size_W,
                           grad_rearranged_mesh_vertex_xy,
                           grad_rearranged_mesh_vertex_texture);
}

