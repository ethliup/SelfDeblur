#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

void forward_flow_renderer(const float* rearranged_mesh_vertex_xy,
                           const float* rearranged_mesh_vertex_texture,
                           const int batchSize,
                           const int nTrianglesPerBatch,
                           const int imageH,
                           const int imageW,
                           const int nMaxTrianglesPerPixel,
                           const int nChannelsTexture,
                           float* bufferBaycentricCoeffs,
                           int* bufferMapPixelToTriangles,
                           float* rendered_image,
                           float* rendered_weight,
                           int *  rendered_face_index,
                           float* rendered_mask);

void backward_flow_renderer(const float* rearranged_mesh_vertex_xy,
                            const float* rearranged_mesh_vertex_texture,
                            const float* rendered_image,
                            const int*   rendered_face_index,
                            const float* rendered_weight,
                            const float* grad_rendered_image,
                            const int nchan_texture,
                            const int batch_size,
                            const int n_triangles_per_batch,
                            const int image_size_H,
                            const int image_size_W,
                            float*  grad_rearranged_mesh_vertex_xy,
                            float*  grad_rearranged_mesh_vertex_texture);

#endif