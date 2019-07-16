#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include <chrono>
#include "nmmintrin.h"
#include <immintrin.h>
#include "common.h"

static float normal_sqrt(float * a, int n)
{
        float sum = 0.0;
        for (int i = 0; i < n; i++)
        {
                sum += sqrt(a[i]);
        }
        return sum;
}

static float normal_sqrt_loop4(float * a, int n)
{
        float sum = 0.0;
        size_t block = n / 4;
        size_t reserve = n % 4;
        float * p = a;
        for (size_t i = 0; i < block; ++i)
        {
                sum += sqrt(p[0]);
                sum += sqrt(p[1]);
                sum += sqrt(p[2]);
                sum += sqrt(p[3]);
                p += 4;
        }

        for (size_t i = 0; i < reserve; ++i)
        {
                sum += sqrt(p[i]);
        }
        return sum;
}

static float sse_sqrt(float * a, int n)
{
        float sum = 0;
        __m128 sse_load;
        const float *p = a;

        int nBlockWidth = 4;
        size_t block = n / nBlockWidth;
        size_t reserved = n % nBlockWidth;

        for (size_t i = 0; i < block; ++i)
        {
                sse_load = _mm_load_ps(p);
                __m128 temp = _mm_sqrt_ps(sse_load);
                const float * q = (const float*)&temp;
                sum += q[0] + q[1] + q[2] + q[3];
                p += nBlockWidth;
        }

        for (size_t i = 0; i < reserved; i++)
        {
                sum += sqrt(p[i]);
        }
        return sum;
}

static float sse_sqrt_256(float * a, int n)
{
        float sum = 0;
        __m256 sse_load;
        const float *p = a;

        int nBlockWidth = 8;
        size_t block = n / nBlockWidth;
        size_t reserved = n % nBlockWidth;

        for (size_t i = 0; i < block; ++i)
        {
                sse_load = _mm256_load_ps(p);
                __m256 temp = _mm256_sqrt_ps(sse_load);
                const float * q = (const float*)&temp;
                sum += q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];
                p += nBlockWidth;
        }

        for (size_t i = 0; i < reserved; i++)
        {
                sum += sqrt(p[i]);
        }
        return sum;
}

int sqrt_test(int argc, char ** argv)
{
        int n = 40000;
        printf("n=%d\n", n);
        float * a = (float*)TAlignAlloc(sizeof(float)*n, 16);
        for (int i = 0; i < n; i++)
        {
                a[i] = (float)(i % 10);
        }

        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;//∫¡√Î
        typedef std::chrono::duration<float> fsec;

        float sum = 0;

        {
                auto t0 = Time::now();
                sum = normal_sqrt(a, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("normal_num,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = normal_sqrt_loop4(a, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("normal_sqrt_loop4,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = sse_sqrt(a, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("sse_sqrt,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = sse_sqrt_256(a, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("sse_sqrt_256,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }
        return 0;
}