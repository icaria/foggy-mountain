__kernel void calculate_forces(__global int *points, __global float4 * globalP, __global float4 * globalA)
{

    int i = get_global_id(0);

    float4 myPosition = globalP[i];
    float4 acc = 0;
    acc.w = 1;

    //for(int k = 0; k < POINTS; k++) { 
    //    body_body(myPosition, globalP[k], &acc);
    //}

    globalA[i] = acc;

}

__kernel void body_body(__global float4 * bi, __global float4 * bj, __global float4 * ai)
{
//    float4 r;
    
//    r.x = bj.x - bi.x;
//    r.y = bj.y - bi.y;
//    r.z = bj.z - bi.z;
//    r.w = 1.0f;

//    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS;

//    float distSixth = distSqr * distSqr * distSqr;
//    float invDistCube = 1.0f/sqrtf(distSixth);

//    float s = bj.w * invDistCube;

//    ai->x += r.x * s;
//    ai->y += r.y * s;
//    ai->z += r.z * s;
}
