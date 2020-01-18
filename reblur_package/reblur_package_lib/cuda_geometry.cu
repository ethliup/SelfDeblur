#include "cuda_geometry.h"
#include <cuda_runtime_api.h>

__global__ void rearrange_vertices_kernel(const float* vertex_array, 
                                          const int* faces, 
                                          const int nVertexPerBatch, 
                                          const int nFacesPerBatch, 
                                          const int batch_size, 
                                          const int vertexDim, 
                                          float* rearranged_vertex_array)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * nFacesPerBatch) {
        return;
    }

    int bn = i / nFacesPerBatch;

    const int* face = &faces[i * 3];
    int v0 = face[0] + nVertexPerBatch * bn;
    int v1 = face[1] + nVertexPerBatch * bn;
    int v2 = face[2] + nVertexPerBatch * bn;

    float* vertex = &rearranged_vertex_array[i * vertexDim * 3];

    memcpy(vertex, vertex_array+v0*vertexDim, vertexDim*sizeof(float));
    memcpy(vertex + vertexDim, vertex_array + v1 * vertexDim, vertexDim * sizeof(float));
    memcpy(vertex + vertexDim * 2, vertex_array + v2 * vertexDim, vertexDim * sizeof(float));
}

__global__ void backward_rearrange_vertices_kernel(const float* grad_rearranged_vertices, 
                                                   const int* faces, 
                                                   const int nVertexPerBatch, 
                                                   const int nFacesPerBatch, 
                                                   const int batch_size, 
                                                   const int vertexDim, 
                                                   const float fraction, 
                                                   float* grad_vertices) 
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * nFacesPerBatch) {
        return;
    }

    int bn = i / nFacesPerBatch;

    const int* face = &faces[i * 3];
    int v0 = face[0] + nVertexPerBatch * bn;
    int v1 = face[1] + nVertexPerBatch * bn;
    int v2 = face[2] + nVertexPerBatch * bn;

    const float* grad_rearranged_vertex = &grad_rearranged_vertices[i * vertexDim * 3];
    
    for(int j=0; j < vertexDim; j++)
    {
        atomicAdd(grad_vertices + v0 * vertexDim + j, grad_rearranged_vertex[j] * fraction);
        atomicAdd(grad_vertices + v1 * vertexDim + j, grad_rearranged_vertex[vertexDim+j] * fraction);
        atomicAdd(grad_vertices + v2 * vertexDim + j, grad_rearranged_vertex[2*vertexDim+j] * fraction);
    }
}

__global__ void kernel_mask_via_flow_forward_warp(const float* flow_forward_warped_xy, int B, int H, int W, float* mask)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= B*H*W) return;

    float xf = flow_forward_warped_xy[2*i];
    float yf = flow_forward_warped_xy[2*i+1];

    if(xf > W-1 || xf < 0 || yf > H-1 || yf < 0) return;
    int tl_x = xf; int tl_y = yf;
    int tr_x = tl_x+1; int tr_y = tl_y;
    int bl_x = tl_x; int bl_y = tl_y + 1;
    int br_x = tl_x + 1; int br_y = tl_y + 1;

    int bn = i/(H*W);

    if(tl_x<W && tl_x>-1 && tl_y<H && tl_y>-1) atomicAdd(mask + bn*H*W + tl_y*W + tl_x, (1.0 - (xf - tl_x)) * (1.0 - (yf - tl_y)));
    if(tr_x<W && tr_x>-1 && tr_y<H && tr_y>-1) atomicAdd(mask + bn*H*W + tr_y*W + tr_x, (1.0 - (tr_x - xf)) * (1.0 - (yf - tl_y)));
    if(bl_x<W && bl_x>-1 && bl_y<H && bl_y>-1) atomicAdd(mask + bn*H*W + bl_y*W + bl_x, (1.0 - (xf - bl_x)) * (1.0 - (bl_y - yf)));
    if(br_x<W && br_x>-1 && br_y<H && br_y>-1) atomicAdd(mask + bn*H*W + br_y*W + br_x, (1.0 - (br_x - xf)) * (1.0 - (br_y - yf)));
}


/**********************************************************************/
/*****                       C++ wrappers                         *****/ 
/**********************************************************************/
void rearrange_vertices(const float* vertex_array, 
                        const int* faces, 
                        const int nVertexPerBatch, 
                        const int nFacesPerBatch, 
                        const int batch_size, 
                        const int vertexDim, 
                        float* new_vertex_array)
{
    const int threads = 512;
    const dim3 blocks ((batch_size * nFacesPerBatch) / threads + 1);
    rearrange_vertices_kernel<<<blocks, threads>>>(vertex_array, faces, nVertexPerBatch, nFacesPerBatch, batch_size, vertexDim, new_vertex_array);
    cudaDeviceSynchronize();
}

void backward_rearrange_vertices(const float* grad_rearranged_vertices, 
                                 const int* faces, 
                                 const int nVertexPerBatch, 
                                 const int nFacesPerBatch, 
                                 const int batch_size, 
                                 const int vertexDim, 
                                 const float fraction, 
                                 float* grad_vertices)
{
    const int threads = 512;
    const dim3 blocks ((batch_size * nFacesPerBatch) / threads + 1);
    backward_rearrange_vertices_kernel<<<blocks, threads>>>(grad_rearranged_vertices, faces, nVertexPerBatch, nFacesPerBatch, batch_size, vertexDim, fraction, grad_vertices);
    cudaDeviceSynchronize();
}

void mask_via_flow_forward_warp(const float* flow_forward_warped_xy, int B, int H, int W, float* mask)
{
  const int nThreads = 512;
  const dim3 blocks((B*H*W) / nThreads + 1);
  kernel_mask_via_flow_forward_warp<<<blocks, nThreads>>>(flow_forward_warped_xy, B, H, W, mask);
  cudaDeviceSynchronize();
}



