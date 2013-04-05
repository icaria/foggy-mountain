__kernel void calculate_forces(__global int points, __global float4 * globalP, __global float4 * globalA)
{

    int i = get_global_id(0);

    float4 myPosition = globalP[i];
    float4 acc = 0;
    acc.w = 1;

    for(int k = 0; k < points; k++) { 
        bodyBodyInteraction(myPosition, globalP[k], &acc);
    }

    globalA[i] = acc;

}

//__kernel void body_body(__global float4 bi, __global float4 bj, __global float4 *ai)
//{


//}
