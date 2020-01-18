#ifndef CUDA_GEOMETRY_H
#define CUDA_GEOMETRY_H

void rearrange_vertices(const float* vertex_array, 
	                    const int* faces, 
	                    const int nVertexPerBatch, 
	                    const int nFacesPerBatch, 
	                    const int batch_size, 
	                    const int vertexDim, 
	                    float* rearranged_vertex_array);

void backward_rearrange_vertices(const float* grad_rearranged_vertices, 
	                             const int* faces, 
	                             const int nVertexPerBatch, 
	                             const int nFacesPerBatch, 
	                             const int batch_size, 
	                             const int vertexDim, 
	                             const float fraction, 
	                             float* grad_vertices);

void mask_via_flow_forward_warp(const float* flow_forward_warped_xy, int B, int H, int W, float* mask);

#endif