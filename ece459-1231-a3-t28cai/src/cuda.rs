// This is the skeleton for the CUDA implementation
use crate::cnn::*;
use rustacuda::function::BlockSize;
use rustacuda::launch;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;
use std::convert::TryFrom;
// Fields need to be ordered this way so the DeviceBoxes are
// dropped before the Context. Otherwise the drop will panic.

pub struct CudaContext {
    conv_layer: DeviceBox<ConvLayer>,
    output_layer: DeviceBox<OutputLayer>,
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaContext {
    pub fn init(cnn: &Cnn) -> Result<Self, Box<dyn Error>> {
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let l_context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        let ptx = CString::new(include_str!("../kernel/kernel.ptx"))?;
        let l_module = Module::load_from_string(&ptx)?;
        let l_stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(CudaContext {
            conv_layer: DeviceBox::new(&cnn.conv_layer).unwrap(),
            output_layer: DeviceBox::new(&cnn.output_layer).unwrap(),
            module: l_module,
            stream: l_stream,
            _context: l_context,
        })
    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        // println!("in compute");
        // for testing purposes, let see if we can just hook up cpu and it works
        let mut output = OutputVec([0.0; OUT_LAYER_SIZE]);
        let mut input_matrix = DeviceBox::new(input).unwrap(); // <-- first
        let mut layer_1_output = DeviceBox::new(&ConvOutput(
            [[[0.0; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE],
        ))
        .unwrap();

        let mut layer_2_output = DeviceBox::new(&OutputVec(
            [0.0; OUT_LAYER_SIZE],
        ))
        .unwrap();
        unsafe {
            // Launch the kernel with one block of one thread, no dynamic shared memory on ‘stream‘.
            let module = &self.module;
            let stream = &self.stream;
            let size: u32 = u32::try_from(CONV_OUT_DIM).unwrap();
            let layer_size: u32 = u32::try_from(CONV_LAYER_SIZE).unwrap();

            let result = launch!(module.convolution_layer<<<(size*size*layer_size)/32, 32, 0, stream>>>(
                input_matrix.as_device_ptr(),
                self.conv_layer.as_device_ptr(),
                layer_1_output.as_device_ptr()
            ));
            result?;
        }
        self.stream.synchronize()?;


        unsafe {
            // Launch the kernel with one block of one thread, no dynamic shared memory on ‘stream‘.
            let module = &self.module;
            let stream = &self.stream;
            let size: u32 = u32::try_from(OUT_LAYER_SIZE).unwrap();

            let result = launch!(module.output_layer<<<size, 1, 0, stream>>>(
                layer_1_output.as_device_ptr(),
                self.output_layer.as_device_ptr(),
                layer_2_output.as_device_ptr()
            ));
            result?;
        }

        self.stream.synchronize()?;

        layer_2_output.copy_to(&mut output);
        Ok(output)
    }
}
