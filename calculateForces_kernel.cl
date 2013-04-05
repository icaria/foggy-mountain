__kernel void calculateForces(__global int global_id, __global cl_float4 * globalP, __global cl_float4 * globalA)
{
    cl_float4 myPosition = globalP[global_id];
    cl_float4 acc = {{0.0f, 0.0f, 0.0f, 1.0f}};

    int id = get_global_id(0);
    bodyBodyInteraction(myPosition, globalP[id], &acc);

    globalA[global_id] = acc;

}
