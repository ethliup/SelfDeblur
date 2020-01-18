#include "cuda_renderer.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

template<typename scalar_t>
__device__ scalar_t sign(scalar_t x0, scalar_t y0, scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2)
{
    return (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2);
}

template<typename scalar_t>
__device__ bool PointInTriangle(scalar_t xp, scalar_t yp, scalar_t x0, scalar_t y0, scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2)
{
    scalar_t d1, d2, d3;
    bool has_neg, has_pos;

    d1 = sign<scalar_t>(xp, yp, x0, y0, x1, y1);
    d2 = sign<scalar_t>(xp, yp, x1, y1, x2, y2);
    d3 = sign<scalar_t>(xp, yp, x2, y2, x0, y0);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}

template <typename scalar_t>
__global__ void preprocess_baycentric_kernel(const scalar_t* rearranged_mesh_vertex_xy,
                                      const int bs,
                                      const int nf,
                                      scalar_t* baycentric_coeffs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bs * nf) return;

    const scalar_t* triangle = &rearranged_mesh_vertex_xy[i*3*2];
    scalar_t* baycentric_coeff = &baycentric_coeffs[i*9];

    scalar_t p[3][2];
    for (int num = 0; num < 3; num++) {
        p[num][0] = triangle[2 * num];
        p[num][1] = triangle[2 * num + 1];
    }

    // compute face_inv
    scalar_t face_inv[9] = {
            p[1][1] - p[2][1], p[2][0] - p[1][0], p[1][0] * p[2][1] - p[2][0] * p[1][1],
            p[2][1] - p[0][1], p[0][0] - p[2][0], p[2][0] * p[0][1] - p[0][0] * p[2][1],
            p[0][1] - p[1][1], p[1][0] - p[0][0], p[0][0] * p[1][1] - p[1][0] * p[0][1]};
    scalar_t face_inv_denominator = (
            p[2][0] * (p[0][1] - p[1][1]) +
            p[0][0] * (p[1][1] - p[2][1]) +
            p[1][0] * (p[2][1] - p[0][1]));
    for (int k = 0; k < 9; k++) {
        face_inv[k] /= face_inv_denominator;
    }

    // set to global memory
    for (int k = 0; k < 9; k++) {
        baycentric_coeff[k] = face_inv[k];
    }
}

template<typename scalar_t>
__global__ void register_triangle_to_pixels_kernel(const scalar_t* rearranged_mesh_vertex_xy,
                                            const int bs,
                                            const int nf,
                                            const int image_H,
                                            const int image_W,
                                            const int nMaxTrianglesPerPixel,
                                            int*  map_pixel_to_triangles)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= bs * nf) return;

    const int bn = i / nf;
    const int fn = i % nf;
    const int is_H = image_H;
    const int is_W = image_W;

    const scalar_t* triangle = &rearranged_mesh_vertex_xy[i * 3 * 2];

    const scalar_t x0=triangle[0];
    const scalar_t y0=triangle[1];

    const scalar_t x1=triangle[2];
    const scalar_t y1=triangle[3];

    const scalar_t x2=triangle[4];
    const scalar_t y2=triangle[5];

    const int x_tl_i = min(min(x0, x1), x2);
    const int x_br_i = max(max(x0, x1), x2);
    const int y_tl_i = min(min(y0, y1), y2);
    const int y_br_i = max(max(y0, y1), y2);


    // iterate over all pixels within bounding box, to check if they are inside triangle
    for(int yi = y_tl_i - 1; yi < y_br_i + 2; yi++)
    {
        for(int xi = x_tl_i - 1; xi < x_br_i + 2; xi++)
        {
            if (xi > is_W-1 || yi > is_H-1 || xi < 0 || yi < 0) continue;

            if (!PointInTriangle<scalar_t>(xi - 1e-5, yi - 1e-4, x0, y0, x1, y1, x2, y2)) continue;

            // register to global memory
            int index = bn * is_H * is_W + yi * is_W + xi;
            int start_index = 0;

            while (atomicCAS(&map_pixel_to_triangles[index * nMaxTrianglesPerPixel + start_index], -1, fn) != -1) {
                start_index = start_index + 1;

                if (start_index == nMaxTrianglesPerPixel) {
                    printf("register faces %d to pixels reaches maximum ...\n", i);
                    break;
                }
            }
        }
    }
}

template<typename scalar_t>
__global__ void forward_flow_renderer_kernel(const scalar_t* rearranged_mesh_vertex_xy,
                                             const scalar_t* rearranged_mesh_vertex_texture,
                                             const scalar_t* baycentric_coeffs,
                                             const int*  map_pixel_to_triangles,
                                             const int bs,
                                             const int n_triangles_per_batch,
                                             const int image_H,
                                             const int image_W,
                                             const int nMaxTrianglesPerPixel,
                                             const int nchan_texture,
                                             scalar_t* rendered_image,
                                             scalar_t* rendered_weight,
                                             int *     rendered_face_index,
                                             float*    rendered_mask)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bs * image_H * image_W) return;

    const int is_H = image_H;
    const int is_W = image_W;
    const int bn = i / (is_H * is_W);
    const int pn = i % (is_H * is_W);
    const int yi = pn / is_W;
    const int xi = pn % is_W;

    const int* fns = &map_pixel_to_triangles[i * nMaxTrianglesPerPixel];
    for(int k = 0; k < nMaxTrianglesPerPixel; k++)
    {
        const int fn = fns[k];
        if (fn == -1) break;

        const scalar_t* triangle_texture = &rearranged_mesh_vertex_texture[bn*n_triangles_per_batch*3*nchan_texture + fn*3*nchan_texture];
        const scalar_t* baycentric_coeff = &baycentric_coeffs[bn*n_triangles_per_batch*9 + fn*9];

        const scalar_t fi00 = baycentric_coeff[0];
        const scalar_t fi01 = baycentric_coeff[1];
        const scalar_t fi02 = baycentric_coeff[2];
        const scalar_t fi10 = baycentric_coeff[3];
        const scalar_t fi11 = baycentric_coeff[4];
        const scalar_t fi12 = baycentric_coeff[5];
        const scalar_t fi20 = baycentric_coeff[6];
        const scalar_t fi21 = baycentric_coeff[7];
        const scalar_t fi22 = baycentric_coeff[8];

        scalar_t w[3];
        w[0] = fi00 * xi + fi01 * yi + fi02;
        w[1] = fi10 * xi + fi11 * yi + fi12;
        w[2] = fi20 * xi + fi21 * yi + fi22;

        for(int j = 0; j < nchan_texture; j++)
        {
            rendered_image[nchan_texture*i+j] = w[0]*triangle_texture[j] 
                                                + w[1]*triangle_texture[j+nchan_texture] 
                                                + w[2]*triangle_texture[j+2*nchan_texture];
        }

        if(rendered_mask != NULL) {
            rendered_mask[i] = 1.0;
        }
        rendered_face_index[i]=fn;

        rendered_weight[3*i]=w[0];
        rendered_weight[3*i+1]=w[1];
        rendered_weight[3*i+2]=w[2];
    }
}

template <typename scalar_t>
__global__ void backward_rgb_map_cuda_kernel(const scalar_t* rearranged_mesh_vertex_xy,
                                            const scalar_t* rearranged_mesh_vertex_texture,
                                            const scalar_t* rendered_image,
                                            const int32_t* rendered_face_index,
                                            const scalar_t* rendered_weight,
                                            const scalar_t*  grad_rendered_image,
                                            const int nchan_texture,
                                            const size_t batch_size,
                                            const size_t n_triangles_per_batch,
                                            const int image_size_H,
                                            const int image_size_W,
                                            scalar_t*  grad_rearranged_mesh_vertex_xy,
                                            scalar_t*  grad_rearranged_mesh_vertex_texture) 
{    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * image_size_H * image_size_W) {
        return;
    }

    const int fn = rendered_face_index[i];
    if (0 <= fn) {
        const int nf = n_triangles_per_batch;
        const int is_H = image_size_H;
        const int is_W = image_size_W;
        const int bn = i / (is_H * is_W);

        const scalar_t* v_xy = &rearranged_mesh_vertex_xy[(bn*nf + fn)*6];
        const scalar_t* v_rgb = &rearranged_mesh_vertex_texture[(bn*nf + fn)*3*nchan_texture];
        const scalar_t* weight = &rendered_weight[i * 3];

        scalar_t* grad_pos = &grad_rearranged_mesh_vertex_xy[(bn*nf+fn)*6];
        scalar_t* grad_color = &grad_rearranged_mesh_vertex_texture[(bn*nf+fn)*3*nchan_texture];

        const scalar_t* grad_rgb = &grad_rendered_image[i*nchan_texture];

        /* derivative wrt rgb */
        for (int k =0; k < 3; k++)
        {
            for(int j=0; j < nchan_texture; j++)
            {
                atomicAdd(&grad_color[nchan_texture * k + j], grad_rgb[j] * weight[k]);                
            }
        }

        /* derivative wrt x, y */
        const scalar_t x0 = v_xy[0]; const scalar_t y0 = v_xy[1];
        const scalar_t x1 = v_xy[2]; const scalar_t y1 = v_xy[3];
        const scalar_t x2 = v_xy[4]; const scalar_t y2 = v_xy[5];

        const int pn = i % (is_H * is_W);
        const int y = pn / is_W;
        const int x = pn % is_W; 

        const scalar_t dD_dx0 = y1 - y2; 
        const scalar_t dD_dy0 = x2 - x1;
        const scalar_t dD_dx1 = y2 - y0;
        const scalar_t dD_dy1 = x0 - x2;
        const scalar_t dD_dx2 = y0 - y1;
        const scalar_t dD_dy2 = x1 - x0;

        const scalar_t dF0_dx1 = y2 - y;
        const scalar_t dF0_dy1 = x - x2;
        const scalar_t dF0_dx2 = y - y1;
        const scalar_t dF0_dy2 = x1 - x;

        const scalar_t dF1_dx0 = y - y2;
        const scalar_t dF1_dy0 = x2 - x;
        const scalar_t dF1_dx2 = y0 - y;
        const scalar_t dF1_dy2 = x - x0;

        const scalar_t dF2_dx0 = y1 - y;
        const scalar_t dF2_dy0 = x - x1;
        const scalar_t dF2_dx1 = y - y0;
        const scalar_t dF2_dy1 = x0 - x;

        const scalar_t D = x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1);

        const scalar_t dw0_dx0 = -weight[0] * dD_dx0 / D;
        const scalar_t dw0_dy0 = -weight[0] * dD_dy0 / D;
        const scalar_t dw0_dx1 = (dF0_dx1 - weight[0] * dD_dx1) / D;
        const scalar_t dw0_dy1 = (dF0_dy1 - weight[0] * dD_dy1) / D;
        const scalar_t dw0_dx2 = (dF0_dx2 - weight[0] * dD_dx2) / D;
        const scalar_t dw0_dy2 = (dF0_dy2 - weight[0] * dD_dy2) / D;

        const scalar_t dw1_dx0 = (dF1_dx0 - weight[1] * dD_dx0) / D;
        const scalar_t dw1_dy0 = (dF1_dy0 - weight[1] * dD_dy0) / D;
        const scalar_t dw1_dx1 = (-weight[1] * dD_dx1) / D;
        const scalar_t dw1_dy1 = (-weight[1] * dD_dy1) / D;
        const scalar_t dw1_dx2 = (dF1_dx2 - weight[1] * dD_dx2) / D;
        const scalar_t dw1_dy2 = (dF1_dy2 - weight[1] * dD_dy2) / D;

        const scalar_t dw2_dx0 = (dF2_dx0 - weight[2] * dD_dx0) / D;
        const scalar_t dw2_dy0 = (dF2_dy0 - weight[2] * dD_dy0) / D;
        const scalar_t dw2_dx1 = (dF2_dx1 - weight[2] * dD_dx1) / D;
        const scalar_t dw2_dy1 = (dF2_dy1 - weight[2] * dD_dy1) / D;
        const scalar_t dw2_dx2 = (-weight[2] * dD_dx2) / D;
        const scalar_t dw2_dy2 = (-weight[2] * dD_dy2) / D;

        for(int j=0; j < nchan_texture; j++)
        {
            scalar_t dIj_dx0 = v_rgb[j] * dw0_dx0 + v_rgb[j+nchan_texture] * dw1_dx0 + v_rgb[j+2*nchan_texture] * dw2_dx0;
            scalar_t dIj_dy0 = v_rgb[j] * dw0_dy0 + v_rgb[j+nchan_texture] * dw1_dy0 + v_rgb[j+2*nchan_texture] * dw2_dy0;
            scalar_t dIj_dx1 = v_rgb[j] * dw0_dx1 + v_rgb[j+nchan_texture] * dw1_dx1 + v_rgb[j+2*nchan_texture] * dw2_dx1;
            scalar_t dIj_dy1 = v_rgb[j] * dw0_dy1 + v_rgb[j+nchan_texture] * dw1_dy1 + v_rgb[j+2*nchan_texture] * dw2_dy1;
            scalar_t dIj_dx2 = v_rgb[j] * dw0_dx2 + v_rgb[j+nchan_texture] * dw1_dx2 + v_rgb[j+2*nchan_texture] * dw2_dx2;
            scalar_t dIj_dy2 = v_rgb[j] * dw0_dy2 + v_rgb[j+nchan_texture] * dw1_dy2 + v_rgb[j+2*nchan_texture] * dw2_dy2;

            atomicAdd(&grad_pos[0], grad_rgb[j] * dIj_dx0);
            atomicAdd(&grad_pos[1], grad_rgb[j] * dIj_dy0);

            atomicAdd(&grad_pos[2], grad_rgb[j] * dIj_dx1);
            atomicAdd(&grad_pos[3], grad_rgb[j] * dIj_dy1);

            atomicAdd(&grad_pos[4], grad_rgb[j] * dIj_dx2);
            atomicAdd(&grad_pos[5], grad_rgb[j] * dIj_dy2);
        }
    }
}

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
                           float* rendered_mask)
{
    const int threads = 512;
    const dim3 blocks_1 ((batchSize * nTrianglesPerBatch - 1) / threads +1);

    preprocess_baycentric_kernel<float><<<blocks_1, threads>>>(
            rearranged_mesh_vertex_xy,
            batchSize,
            nTrianglesPerBatch,
            bufferBaycentricCoeffs);
    cudaDeviceSynchronize();

    register_triangle_to_pixels_kernel<float><<<blocks_1, threads>>>(
            rearranged_mesh_vertex_xy,
            batchSize,
            nTrianglesPerBatch,
            imageH,
            imageW,
            nMaxTrianglesPerPixel,
            bufferMapPixelToTriangles);
    cudaDeviceSynchronize();

    const dim3 blocks_3 ((batchSize * imageH * imageW - 1) / threads +1);
    forward_flow_renderer_kernel<float><<<blocks_3, threads>>>(
            rearranged_mesh_vertex_xy,
            rearranged_mesh_vertex_texture,
            bufferBaycentricCoeffs,
            bufferMapPixelToTriangles,
            batchSize,
            nTrianglesPerBatch,
            imageH,
            imageW,
            nMaxTrianglesPerPixel,
            nChannelsTexture,
            rendered_image,
            rendered_weight,
            rendered_face_index,
            rendered_mask);
    cudaDeviceSynchronize();
}

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
                            float*  grad_rearranged_mesh_vertex_texture) 
{
    const int threads = 512;
    const dim3 blocks ((batch_size * image_size_H * image_size_W - 1) / threads + 1);

    backward_rgb_map_cuda_kernel<float><<<blocks, threads>>>(rearranged_mesh_vertex_xy,
        rearranged_mesh_vertex_texture,
          rendered_image,
          rendered_face_index,
          rendered_weight,
          grad_rendered_image,
          nchan_texture,
          batch_size,
          n_triangles_per_batch,
          image_size_H, 
          image_size_W,
          grad_rearranged_mesh_vertex_xy,
          grad_rearranged_mesh_vertex_texture);

    cudaDeviceSynchronize();
}

