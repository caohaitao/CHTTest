#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include <chrono>
#include "nmmintrin.h"
#include <immintrin.h>
#include "common.h"

static float normal_mul(float * a, float * b, int n)
{
        float sum = 0.0;
        for (int i = 0; i < n; i++)
        {
                sum += a[i] * b[i];
        }
        return sum;
}

static float normal_mul_loop4(float * a, float *b, int n)
{
        float sum = 0.0;
        size_t block = n / 4;
        size_t reserve = n % 4;
        float * p = a;
        float * p2 = b;
        for (size_t i = 0; i < block; ++i)
        {
                sum += p[0] * p2[0];
                sum += p[1] * p2[1];
                sum += p[2] * p2[2];
                sum += p[3] * p2[3];
                p += 4;
                p2 += 4;
        }

        for (size_t i = 0; i < reserve; ++i)
        {
                sum += p[i] * p2[i];
        }
        return sum;
}

static float sse_mul(float * a, float * b, int n)
{
        float sum = 0;
        __m128 sse_load;
        __m128 sse_load2;
        const float *p = a;
        const float *p2 = b;

        int nBlockWidth = 4;
        size_t block = n / nBlockWidth;
        size_t reserved = n % nBlockWidth;

        for (size_t i = 0; i < block; ++i)
        {
                sse_load = _mm_load_ps(p);
                sse_load2 = _mm_load_ps(p2);
                __m128 temp = _mm_mul_ps(sse_load, sse_load2);
                const float * q = (const float*)&temp;
                sum += q[0] + q[1] + q[2] + q[3];
                p += nBlockWidth;
                p2 += nBlockWidth;
        }

        for (size_t i = 0; i < reserved; i++)
        {
                sum += p[i] * p2[i];
        }
        return sum;
}

static float sse_mul_loop4(float * a, float * b, int n)
{
        float sum = 0;
        __m128 sse_load[4];
        __m128 sse_load2[4];
        const float *p = a;
        const float *p2 = b;

        int nBlockWidth = 4*4;
        size_t block = n / nBlockWidth;
        size_t reserved = n % nBlockWidth;

        for (size_t i = 0; i < block; ++i)
        {
                sse_load[0] = _mm_load_ps(p);
                sse_load[1] = _mm_load_ps(p+4);
                sse_load[2] = _mm_load_ps(p+8);
                sse_load[3] = _mm_load_ps(p+12);
                sse_load2[0] = _mm_load_ps(p2);
                sse_load2[1] = _mm_load_ps(p2+4);
                sse_load2[2] = _mm_load_ps(p2+8);
                sse_load2[3] = _mm_load_ps(p2+12);
                __m128 temp = _mm_mul_ps(sse_load[0], sse_load2[0]);
                __m128 temp1 = _mm_mul_ps(sse_load[1], sse_load2[1]);
                __m128 temp2 = _mm_mul_ps(sse_load[2], sse_load2[2]);
                __m128 temp3 = _mm_mul_ps(sse_load[3], sse_load2[3]);
                const float * q = (const float*)&temp;
                const float * q1 = (const float*)&temp1;
                const float * q2 = (const float*)&temp2;
                const float * q3 = (const float*)&temp3;
                sum += q[0] + q[1] + q[2] + q[3];
                sum += q1[0] + q1[1] + q1[2] + q1[3];
                sum += q2[0] + q2[1] + q2[2] + q2[3];
                sum += q3[0] + q3[1] + q3[2] + q3[3];
                p += nBlockWidth;
                p2 += nBlockWidth;
        }

        for (size_t i = 0; i < reserved; i++)
        {
                sum += p[i] * p2[i];
        }
        return sum;

}

static float sse_256_mul(float * a, float * b, int n)
{
        float sum = 0;
        __m256 sse_load;
        __m256 sse_load2;
        const float *p = a;
        const float *p2 = b;

        size_t block = n / 8;
        size_t reserved = n % 8;

        for (size_t i = 0; i < block; ++i)
        {
                sse_load = _mm256_load_ps(p);
                sse_load2 = _mm256_load_ps(p2);
                __m256 temp = _mm256_mul_ps(sse_load, sse_load2);
                const float * q = (const float*)&temp;
                sum += q[0] + q[1] + q[2] + q[3]+ q[4] + q[5] + q[6] + q[7];
                p += 8;
                p2 += 8;
        }

        for (size_t i = 0; i < reserved; i++)
        {
                sum += p[i] * p2[i];
        }
        return sum;
}

static float sse_256_mul_loop4(float * a, float * b, int n)
{
        float sum = 0;
        __m256 sse_load[4];
        __m256 sse_load2[4];
        const float *p = a;
        const float *p2 = b;

        int nBlock = 8 * 4;
        size_t block = n / nBlock;
        size_t reserved = n % nBlock;

        for (size_t i = 0; i < block; ++i)
        {
                sse_load[0] = _mm256_load_ps(p);
                sse_load[1] = _mm256_load_ps(p+8);
                sse_load[2] = _mm256_load_ps(p+16);
                sse_load[3] = _mm256_load_ps(p+24);
                sse_load2[0] = _mm256_load_ps(p2);
                sse_load2[1] = _mm256_load_ps(p2+8);
                sse_load2[2] = _mm256_load_ps(p2+16);
                sse_load2[3] = _mm256_load_ps(p2+24);
                __m256 temp = _mm256_mul_ps(sse_load[0], sse_load2[0]);
                __m256 temp1 = _mm256_mul_ps(sse_load[1], sse_load2[1]);
                __m256 temp2 = _mm256_mul_ps(sse_load[2], sse_load2[2]);
                __m256 temp3 = _mm256_mul_ps(sse_load[3], sse_load2[3]);
                const float * q = (const float*)&temp;
                const float * q1 = (const float*)&temp1;
                const float * q2 = (const float*)&temp2;
                const float * q3 = (const float*)&temp3;
                sum += q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];
                sum += q1[0] + q1[1] + q1[2] + q1[3] + q1[4] + q1[5] + q1[6] + q1[7];
                sum += q2[0] + q2[1] + q2[2] + q2[3] + q2[4] + q2[5] + q2[6] + q2[7];
                sum += q3[0] + q3[1] + q3[2] + q3[3] + q3[4] + q3[5] + q3[6] + q3[7];
                p += nBlock;
                p2 += nBlock;
        }

        for (size_t i = 0; i < reserved; i++)
        {
                sum += p[i] * p2[i];
        }
        return sum;
}

int float_mul_test(int argc, char ** argv)
{
        int n = 40000;
        printf("n=%d\n", n);
        float * a = (float*)TAlignAlloc(sizeof(float)*n, 16);
        for (int i = 0; i < n; i++)
        {
                a[i] = (float)(i % 10);
        }

        float * b = (float*)TAlignAlloc(sizeof(float)*n, 16);
        for (int i = 0; i < n; i++)
        {
                b[i] = (float)(i % 10);
        }

        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;//∫¡√Î
        typedef std::chrono::duration<float> fsec;

        float sum = 0;

        {
                auto t0 = Time::now();
                sum = normal_mul(a, b, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("normal_num,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = normal_mul_loop4(a, b, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("normal_mul_loop4,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = sse_mul(a, b, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("sse_mul,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = sse_mul_loop4(a, b, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("sse_mul_loop4,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = sse_256_mul(a, b, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("sse_256_mul,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = sse_256_mul_loop4(a, b, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("sse_256_mul_loop4,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }
        return 0;
}