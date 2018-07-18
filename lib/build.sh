CUDA_ARCH="-gencode arch=compute_37,code=sm_37"


# Build NMS
cd nms/src/cuda
echo "Compiling nms kernels by nvcc..."
/cm/shared/apps/cuda80/toolkit/8.0.44/bin/nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../../
python build.py
cd ../
