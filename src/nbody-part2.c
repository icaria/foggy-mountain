/* nbody simulation, version 0 */
/* Modified by Patrick Lam; original source: GPU Gems, Chapter 31 */

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>

#define EPS 1e-10

/// runtime on my 2011 computer: 1m; in 2013, 27s.
// on my 2011 laptop, 1m34s
#define POINTS 500 * 64
#define SPACE 1000.0f
#define BINS 1000
#define BINS_PER_DIM 10

cl_float4 * initializePositions() {
    cl_float4 * pts = (cl_float4 *)malloc(sizeof(cl_float4)*POINTS);
    int i;

    srand(42L); // for deterministic results

    for (i = 0; i < POINTS; i++) {
        // quick and dirty generation of points
        // not random at all, but I don't care.
        pts[i].x = ((float)rand())/RAND_MAX * SPACE;
        pts[i].y = ((float)rand())/RAND_MAX * SPACE;
        pts[i].z = ((float)rand())/RAND_MAX * SPACE;
        pts[i].w = 1.0f; // size = 1.0f for simplicity.
    }
    return pts;
}

cl_float4 * initializeAccelerations() {
    cl_float4 * pts = (cl_float4 *)malloc(sizeof(cl_float4)*POINTS);
    int i;

    for (i = 0; i < POINTS; i++) {
        pts[i].x = pts[i].y = pts[i].z = pts[i].w = 0;
    }
    return pts;
}

cl_float4 * initializeBinPositions() {
    cl_float4 * pts = (cl_float4 *)malloc(sizeof(cl_float4)*BINS);
    int i, j, k;
    
    for (i = 0; i < BINS_PER_DIM; i++) {
        for (j = 0; j < BINS_PER_DIM; j++) {
            for (k = 0; k < BINS_PER_DIM; k++) {
                int pIndex = i * 100 + j * 10 + k;
                pts[pIndex].x = k * 100.0f;
                pts[pIndex].y = j * 100.0f;
                pts[pIndex].z = i * 100.0f;
                pts[pIndex].w = 0.0f;
            }
        }
    }
    return pts;
}

cl_float4 * initializeBins() {
    cl_float4 * pts = (cl_float4 *)malloc(sizeof(cl_float4)*BINS);
    int i;
    
    for (i = 0; i < BINS; i++) {
        pts[i].x = pts[i].y = pts[i].z = pts[i].w = 0.0f;
    }
    return pts;
}

int main(int argc, char ** argv)
{
    // Pass an int * to the function
    int *points = ((int *)malloc(sizeof(int)));
    *points = POINTS;
    
    cl_float4 * x = initializePositions();
    cl_float4 * a = initializeAccelerations();
    cl_float4 * binPos = initializeBinPositions();
    cl_float4 * cm = initializeBins();
    cl::Program program;
    std::vector<cl::Device> devices;

    try { 
        // Get available platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        // Select the default platform and create a context using this platform and the GPU
        cl_context_properties cps[3] = { 
            CL_CONTEXT_PLATFORM, 
            (cl_context_properties)(platforms[0])(), 
            0 
        };
        cl::Context context(CL_DEVICE_TYPE_GPU, cps);
 
        // Get a list of devices on this platform
        devices = context.getInfo<CL_CONTEXT_DEVICES>();
 
        // Create a command queue and use the first device
        cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);
        //cl::CommandQueue queue_bin = cl::CommandQueue(context, devices[0]);
 
        // Read source file
        std::ifstream sourceFile("src/calculate_forces_part2_kernel.cl");
        std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

        // Make program of the source code in the context
        program = cl::Program(context, source);
 
        // Build program for these specific devices
        program.build(devices);
 
        // Make kernel
        cl::Kernel kernel(program, "calculate_forces");
        cl::Kernel kernel_bin(program, "bin_points");
        
        // Memory buffers for centre of mass bin function
        cl::Buffer bufferD = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
        cl::Buffer bufferE = cl::Buffer(context, CL_MEM_READ_ONLY, POINTS * sizeof(cl_float4));
        cl::Buffer bufferF = cl::Buffer(context, CL_MEM_READ_ONLY, BINS * sizeof(cl_float4));
        cl::Buffer bufferG = cl::Buffer(context, CL_MEM_READ_WRITE, BINS * sizeof(cl_float4));
        
        // Copy the points, bin coordinates, and centre of mass to memory buffers
        queue.enqueueWriteBuffer(bufferD, CL_TRUE, 0, sizeof(int), points);
        queue.enqueueWriteBuffer(bufferE, CL_TRUE, 0, POINTS * sizeof(cl_float4), x);
        queue.enqueueWriteBuffer(bufferF, CL_TRUE, 0, BINS * sizeof(cl_float4), binPos);
        queue.enqueueWriteBuffer(bufferG, CL_TRUE, 0, BINS * sizeof(cl_float4), cm);
        
        // Set bin arguments to kernel
        kernel_bin.setArg(0, bufferD);
        kernel_bin.setArg(1, bufferE);
        kernel_bin.setArg(2, bufferF);
        kernel_bin.setArg(3, bufferG);
 
        // Run the kernel on specific ND range
        cl::NDRange global_bin(POINTS);
        cl::NDRange local_bin(1);
        queue.enqueueNDRangeKernel(kernel_bin, cl::NullRange, global_bin, local_bin);
        
        // Read the bins' center of mass into a local list
        queue.enqueueReadBuffer(bufferG, CL_TRUE, 0, BINS * sizeof(cl_float4), cm);
        queue.finish();
        
        for (int i = 0; i < BINS; i++)
            printf("(%2.2f,%2.2f,%2.2f,%2.2f)\n",
                   cm[i].x, cm[i].y, cm[i].z, cm[i].w);
        
        
        // Create memory buffers
        cl::Buffer bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
        cl::Buffer bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, POINTS * sizeof(cl_float4));
        cl::Buffer bufferC = cl::Buffer(context, CL_MEM_READ_WRITE, POINTS * sizeof(cl_float4));
        
        // Copy lists A and B to the memory buffers
        queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, sizeof(int), points);
        queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, POINTS * sizeof(cl_float4), x);
        queue.enqueueWriteBuffer(bufferC, CL_TRUE, 0, POINTS * sizeof(cl_float4), a);
        
        // Set interesction arguments to kernel
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        
        // Run the kernel on specific ND range
        cl::NDRange global(POINTS);
        cl::NDRange local(1);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
        
 
        // Read buffer C into a local list
        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, POINTS * sizeof(cl_float4), a);
 
        /*for (int i = 0; i < POINTS; i++)
            printf("(%2.2f,%2.2f,%2.2f,%2.2f) (%2.3f,%2.3f,%2.3f)\n", 
                    x[i].x, x[i].y, x[i].z, x[i].w,
                    a[i].x, a[i].y, a[i].z);*/

    } catch(cl::Error error) {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        if (error.err() != CL_SUCCESS) {
            std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
        }
    }

    free(x);
    free(a);
    free(points);
    return 0;
}
