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
#define SPACE 1000.0f;

void bodyBodyInteraction(cl_float4 bi, cl_float4 bj, cl_float4 *ai) {
    cl_float4 r;
    
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    r.w = 1.0f;

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS;

    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f/sqrtf(distSixth);

    float s = bj.w * invDistCube;

    ai->x += r.x * s;
    ai->y += r.y * s;
    ai->z += r.z * s;
}

void calculateForces(int points, int global_id, cl_float4 * globalP, cl_float4 * globalA) {
    cl_float4 myPosition = globalP[global_id];
    int i;

    cl_float4 acc = {{0.0f, 0.0f, 0.0f, 1.0f}};
    
    for (i = 0; i < points; i ++) {
        bodyBodyInteraction(myPosition, globalP[i], &acc);
    }
    globalA[global_id] = acc;
}

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

int main(int argc, char ** argv)
{
    cl_float4 * x = initializePositions();
    cl_float4 * a = initializeAccelerations();
    cl::Program program;
    std::vector<cl::Device> devices;    

    int *global_id = new int[POINTS];
    for(int i = 0; i < POINTS; i++) {
        global_id[i] = i;
    }

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
 
        // Read source file
        std::ifstream sourceFile("src/calculate_forces_kernel.cl");
        std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

        // Make program of the source code in the context
        program = cl::Program(context, source);
 
        // Build program for these specific devices
        program.build(devices);
 
        // Make kernel
        cl::Kernel kernel(program, "calculate_forces");
 
        // Create memory buffers
        cl::Buffer bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, POINTS * sizeof(cl_float4));
        cl::Buffer bufferB = cl::Buffer(context, CL_MEM_READ_WRITE, POINTS * sizeof(cl_float4));
 
        // Copy lists A and B to the memory buffers
        queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, POINTS * sizeof(cl_float4), x);
        queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, POINTS * sizeof(cl_float4), a);
 
        // Set arguments to kernel
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
 
        // Run the kernel on specific ND range
        cl::NDRange global(POINTS);
        cl::NDRange local(1);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
 
        // Read buffer C into a local list
        queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, POINTS * sizeof(cl_float4), a);
 
        for (int i = 0; i < POINTS; i++)
            printf("(%2.2f,%2.2f,%2.2f,%2.2f) (%2.3f,%2.3f,%2.3f)\n", 
                    x[i].x, x[i].y, x[i].z, x[i].w,
                    a[i].x, a[i].y, a[i].z);

    } catch(cl::Error error) {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        if (error.err() != CL_SUCCESS) {
            std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
        }
    }

    free(x);
    free(a);
    return 0;
}
