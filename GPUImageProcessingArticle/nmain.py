import cv2
import numpy as np
import pyopencl as cl
import subprocess
import threading
import time

class MorphOp:
    "operation types"
    DILATE = 0
    ERODE = 1

def get_gpu_utilization():
    "Continuously fetch GPU utilization and print it in real-time."
    try:
        while True:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            utilization = result.stdout.strip() if result.stdout else "N/A"
            print(f"GPU Utilization: {utilization}%")
            time.sleep(1)  # Update every second
    except Exception as e:
        print(f"Error fetching GPU utilization: {e}")

def applyMorphOp(imgIn, op):
    "apply morphological operation to image using GPU"
    
    # (1) setup OpenCL
    platforms = cl.get_platforms()
    platform = platforms[0]
    devices = platform.get_devices(cl.device_type.GPU)
    device = devices[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context, device, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # (2) allocate memory
    shape = imgIn.T.shape
    imgOut = np.empty_like(imgIn)    

    imgInBuf = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape=shape)  
    imgOutBuf = cl.Image(context, cl.mem_flags.WRITE_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape=shape)  

    # (3) compile OpenCL kernel
    program = cl.Program(context, open('kernel.cl').read()).build()
    kernel = cl.Kernel(program, 'morphOpKernel')  
    kernel.set_arg(0, imgInBuf)  
    kernel.set_arg(1, np.uint32(op))  
    kernel.set_arg(2, imgOutBuf)  

    # (4) Execute kernel with profiling
    cl.enqueue_copy(queue, imgInBuf, imgIn, origin=(0, 0), region=shape, is_blocking=False)  

    event = cl.enqueue_nd_range_kernel(queue, kernel, shape, None)  
    event.wait()
    elapsed_time = (event.profile.end - event.profile.start) * 1e-6  # Convert ns to ms
    print(f"GPU Kernel Execution Time: {elapsed_time:.3f} ms")

    cl.enqueue_copy(queue, imgOut, imgOutBuf, origin=(0, 0), region=shape, is_blocking=True)  
    
    return imgOut

def main():
    "test implementation: read file 'newimg.png', apply dilation and erosion, write output images"
    
    # Start GPU utilization logging in a separate thread
    gpu_thread = threading.Thread(target=get_gpu_utilization, daemon=True)
    gpu_thread.start()
    
    # Read image
    img = cv2.imread('openname.png', cv2.IMREAD_GRAYSCALE)
    
    # Check if image is loaded successfully
    if img is None:
        print("Error: Unable to load image 'newimg.png'")
        return
    
    # Dilate
    dilate = applyMorphOp(img, MorphOp.DILATE)
    cv2.imwrite('dialated_letteropenname.png', dilate)
    
    # Erode
    erode = applyMorphOp(img, MorphOp.ERODE)
    cv2.imwrite('eroded_letteropenname.png', erode)

if __name__ == '__main__':
    main()