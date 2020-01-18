#include "corr_cuda.h"
#include "corr_cuda_kernel.h"

corr_cuda::corr_cuda()
{}

void corr_cuda::forward(at::Tensor input1,
                      at::Tensor input2,
                      at::Tensor rbot1,
                      at::Tensor rbot2,
                      at::Tensor output,
                      int pad_size,
                      int kernel_size,
                      int max_displacement,
                      int stride1,
                      int stride2,
                      int corr_type_multiply,
                      int batchSize,
                      int nInputPlane,
                      int nInputRows,
                      int nInputCols
                      )
{
    int inputWidthHeight = nInputRows * nInputCols;

    int kernel_radius_ = (kernel_size - 1) / 2;
    int border_size_ = max_displacement + kernel_radius_; // size of unreachable border region (on each side)

    int paddedbottomheight = nInputRows + 2 * pad_size;
    int paddedbottomwidth = nInputCols + 2 * pad_size;

    int nOutputCols = ceil((float)(paddedbottomwidth - border_size_ * 2) / (float)stride1);
    int nOutputRows = ceil((float)(paddedbottomheight - border_size_ * 2) / (float)stride1);

    // Given a center position in image 1, how many displaced positions in -x / +x
    // direction do we consider in image2 (neighborhood_grid_width)
    int neighborhood_grid_radius_ = max_displacement / stride2;
    int neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;

    // Number of output channels amounts to displacement combinations in X and Y direction
    int nOutputPlane = neighborhood_grid_width_ * neighborhood_grid_width_;

    // Inputs
    float * input1_data = input1.data<float>();
    float * input2_data = input2.data<float>();

    // Outputs
    float * output_data = output.data<float>();

    float * rbot1_data = rbot1.data<float>();
    float * rbot2_data = rbot2.data<float>();

    int pwidthheight = paddedbottomwidth * paddedbottomheight;

    blob_rearrange_ongpu(input1_data,rbot1_data,batchSize,nInputPlane,nInputCols,nInputRows,inputWidthHeight,pad_size,pwidthheight);

    blob_rearrange_ongpu(input2_data,rbot2_data,batchSize,nInputPlane,nInputCols,nInputRows,inputWidthHeight,pad_size,pwidthheight);

    CorrelateData_ongpu(rbot1_data,rbot2_data,output_data,batchSize,nOutputCols,nOutputRows,nOutputPlane,max_displacement,neighborhood_grid_radius_,neighborhood_grid_width_,kernel_radius_,kernel_size,stride1,stride2,paddedbottomwidth,paddedbottomheight,nInputPlane,corr_type_multiply);
}

void corr_cuda::backward(at::Tensor input1,
                        at::Tensor input2,
                        at::Tensor rbot1,
                        at::Tensor rbot2,
                        at::Tensor gradOutput,
                        at::Tensor gradInput1,
                        at::Tensor gradInput2,
                        int pad_size,
                        int kernel_size,
                        int max_displacement,
                        int stride1,
                        int stride2,
                        int corr_type_multiply,
                        int batchSize,
                        int nInputPlane,
                        int nInputRows,
                        int nInputCols)
{

    float * input1_data = input1.data<float>();
    float * input2_data = input2.data<float>();

    float * gradOutput_data = gradOutput.data<float>();
    float * gradInput1_data = gradInput1.data<float>();
    float * gradInput2_data = gradInput2.data<float>();

    int inputWidthHeight = nInputRows * nInputCols;

    int kernel_radius_ = (kernel_size - 1) / 2;
    int border_size_ = max_displacement + kernel_radius_; // size of unreachable border region (on each side)

    int paddedbottomheight = nInputRows + 2 * pad_size;
    int paddedbottomwidth = nInputCols + 2 * pad_size;

    int nOutputCols = ceil((float)(paddedbottomwidth - border_size_ * 2) / (float)stride1);
    int nOutputRows = ceil((float)(paddedbottomheight - border_size_ * 2) / (float)stride1);

    // Given a center position in image 1, how many displaced positions in -x / +x
    // direction do we consider in image2 (neighborhood_grid_width)
    int neighborhood_grid_radius_ = max_displacement / stride2;
    int neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;

    // Number of output channels amounts to displacement combinations in X and Y direction
    int nOutputPlane = neighborhood_grid_width_ * neighborhood_grid_width_;

    float * rbot1_data = rbot1.data<float>();
    float * rbot2_data = rbot2.data<float>();

    int pwidthheight = paddedbottomwidth * paddedbottomheight;

    blob_rearrange_ongpu(input1_data,rbot1_data,batchSize,nInputPlane,nInputCols,nInputRows,inputWidthHeight,pad_size,pwidthheight);

    blob_rearrange_ongpu(input2_data,rbot2_data,batchSize,nInputPlane,nInputCols,nInputRows,inputWidthHeight,pad_size,pwidthheight);

    // CorrelationLayerBackward

    CorrelateDataBackward_ongpu(rbot1_data,rbot2_data,gradOutput_data,gradInput1_data,gradInput2_data,batchSize,nOutputCols,nOutputRows,nOutputPlane,max_displacement,neighborhood_grid_radius_,neighborhood_grid_width_,kernel_radius_,stride1,stride2,nInputCols,nInputRows,paddedbottomwidth,paddedbottomheight,nInputPlane,pad_size,corr_type_multiply);
}
