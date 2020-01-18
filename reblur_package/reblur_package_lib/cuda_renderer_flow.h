#ifndef __CUDA_RENDER_FLOW_HEADER__
#define __CUDA_RENDER_FLOW_HEADER__

// Include standard headers
#include "cuda_tensor.h"

class Cuda_renderer_flow
{
public:
    Cuda_renderer_flow(int nChannelsTexture, int H, int W, int nTrianglesPerBatch, int nBatches);
    ~Cuda_renderer_flow();

public:
    void forward(const float* rearranged_mesh_vertex_xy,
                 const float* rearranged_mesh_vertex_texture,
                 float* rendered_image,
                 float* rendered_mask,
                 float* rendered_weight,
                 int*   rendered_face_index);

    void backward(const float* rearranged_mesh_vertex_xy,
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
                  float*  grad_rearranged_mesh_vertex_texture);

protected:
    int m_image_H;
    int m_image_W;
    int m_nTrianglesPerBatch;
    int m_nBatches;
    int m_nMaxTrianglesperPixel;
    int m_nChannelTexture;

    CudaTensor<float>* m_buffer_baycentric_coeffs;
    CudaTensor<int>*   m_buffer_map_pixel_to_triangles;
};

#endif



