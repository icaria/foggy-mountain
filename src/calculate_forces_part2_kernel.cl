#define EPS 1e-10

__kernel void calculate_forces(__global int * points, __global float4 * globalP, __global float4 * globalA)
{
    
    int i = get_global_id(0);
    int k;

    float4 myPosition = globalP[i];
    float4 acc = 0;
    acc.w = 1;

    for(k = 0; k < *points; k++) { 
        float4 r;
        float4 bj = globalP[k];
    
        r.x = bj.x - myPosition.x;
        r.y = bj.y - myPosition.y;
        r.z = bj.z - myPosition.z;
        r.w = 1.0f;

        float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS;

        float distSixth = distSqr * distSqr * distSqr;
        float invDistCube = 1.0f/sqrt(distSixth);

        float s = bj.w * invDistCube;

        acc.x += r.x * s;
        acc.y += r.y * s;
        acc.z += r.z * s;
    }

    globalA[i] = acc;

}

__kernel void bin_points(__global int * points, __global float4 * pts, __global float4 * bins, __global float4 * cm)
{
    int j = get_global_id(0);
    int i;
    
    float4 binPos = bins[j];
    float4 centre = 0;
    
    for (i = 0; i < *points; i++) {
        float4 point = pts[i];
        if (point.x >= binPos.x && point.x < binPos.x + 100.0f &&
            point.y >= binPos.y && point.y < binPos.y + 100.0f &&
            point.z >= binPos.z && point.z < binPos.z + 100.0f   ) {
            centre.x += point.x;
            centre.y += point.y;
            centre.z += point.z;
            centre.w += 1.0f;
        }
    }
    centre.x /= centre.w;
    centre.y /= centre.w;
    centre.z /= centre.w;
    
    cm[j] = centre;
    
}