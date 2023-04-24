
# Pull Request
GPU optimization for parallelized execution of convolutional neural network
# Summary
Introduced cudo parallization of the existing CNN code, convolution_layer and relu layer is merged into a single job, split into 125 blocks each with 32 task, and lastly the code for output_layer is split into 10 different task, each merge 4000 results into the corresponding output


It was verified that the output result of such network is identical to the cpu version
# Tech Detail

During initialization, we setup the gpu driver such that it we can use the first avaialble gpu and loaded in the corresponding symbols from kernel.ptx. once that is done, we pass the created CudaContext back to main.

Later, main invoke the cudacontext to compute the neural network, which goes through 2 layers of processing, first layer handles the relu and convolution, while the later handles the output_layer. At first I thought about having the code also produce to output_layer, but I realized that means we might have a race condition when it comes to data access, thus, I decided to split them into 2 seperate calls to eliminate source of error

I decided to merge relu and convolution layer together since the output can be directly evaluated together, this is easier to manage and overall faster than output then recopy again. I could have also make the output_layer more parallel, but the state it is already had a significant speed up comare to the cpu version.

# Correctness
In order to verify correctness, I decided to split the work into different phases. I first ensure the initialization is correct and code is able to reach compute, then I started to produce the logic of each layer one by one and compare each layer output one by one. There is a commented out code in main.rs which flattens out the ConvOutput and allow compare.py to check correctness. Through this approach, I was able to ensure every bits of each layer is correct, all the way to the final output.

# Performance
Compare to the cpu approach which averages about 30000 microseconds of actual work, the gpu approach typically lies around 10000 microseconds. This is not to say my approach is perfect or anything (in fact, there is many room of improvement and I think I can cut at least abother 2000 microseconds off), but it is a drastic improvement from the sequential version of the code.