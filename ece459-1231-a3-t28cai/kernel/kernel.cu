// Very minimal skeleton for the kernel

#include <stdio.h>
extern "C" __constant__ int INPUT_DIM = 100;
extern "C" __constant__ int FILTER_DIM = 5;
extern "C" __constant__ int CONV_LAYER_SIZE = 10;
extern "C" __constant__ int CONV_OUTPUT_DIM = 20;

// used for test compilation
extern "C" __global__ void sum(const float *x, const float *y, float *out, int count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
    {
        out[i] = x[i] + y[i];
    }
}

extern "C" __global__ void convolution_layer(const double *input, const double *conv_filters, double *outputs)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int layer = index/ (CONV_OUTPUT_DIM*CONV_OUTPUT_DIM);
    int i = ((index % (CONV_OUTPUT_DIM * CONV_OUTPUT_DIM)) / (CONV_OUTPUT_DIM)) * 5;
    int j = (index% (CONV_OUTPUT_DIM)) * 5;

    for (int x = 0; x < FILTER_DIM; x++)
    {
        for (int y = 0; y < FILTER_DIM; y++)
        {
            outputs[index] += input[(i + x) * (INPUT_DIM) + (j + y)] * conv_filters[layer*FILTER_DIM*FILTER_DIM + x * FILTER_DIM + y];
        }
    }

    if (outputs[index] < 0.0) {
        outputs[index] = 0.0;
    }
}

extern "C" __global__ void output_layer(const double *input, const double *weight, double *outputs)
{   
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    for (int x = 0; x < CONV_OUTPUT_DIM * CONV_OUTPUT_DIM * CONV_LAYER_SIZE; x++)
    {
        outputs[index] += input[x] * weight[CONV_OUTPUT_DIM * CONV_OUTPUT_DIM * CONV_LAYER_SIZE * index + x];
    }
}