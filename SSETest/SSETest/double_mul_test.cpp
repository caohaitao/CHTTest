#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include <chrono>
#include "nmmintrin.h"
#include <immintrin.h>
#include "common.h"

static double normal_mul(double * a, double * b, int n)
{
        double sum = 0.0;
        for (int i = 0; i < n; i++)
        {
                sum += a[i] * b[i];
        }
        return sum;
}

static double normal_mul_loop4(double * a, double *b, int n)
{
        double sum = 0.0;
        size_t block = n / 4;
        size_t reserve = n % 4;
        double * p = a;
        double * p2 = b;
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

static double sse_mul(double * a, double * b, int n)
{
        double sum = 0;
        __m128d sse_load;
        __m128d sse_load2;
        const double *p = a;
        const double *p2 = b;

        int nBlockWidth = 2;
        size_t block = n / nBlockWidth;
        size_t reserved = n % nBlockWidth;

        for (size_t i = 0; i < block; ++i)
        {
                sse_load = _mm_load_pd(p);
                sse_load2 = _mm_load_pd(p2);
                __m128d temp = _mm_mul_pd(sse_load, sse_load2);
                const double * q = (const double*)&temp;
                sum += q[0] + q[1];
                p += nBlockWidth;
                p2 += nBlockWidth;
        }

        for (size_t i = 0; i < reserved; i++)
        {
                sum += p[i] * p2[i];
        }
        return sum;
}

static double sse_mul_loop4(double * a, double * b, int n)
{
        double sum = 0;
        __m128d sse_load[4];
        __m128d sse_load2[4];
        const double *p = a;
        const double *p2 = b;

        int nBlockWidth = 2 * 4;
        size_t block = n / nBlockWidth;
        size_t reserved = n % nBlockWidth;

        for (size_t i = 0; i < block; ++i)
        {
                sse_load[0] = _mm_load_pd(p);
                sse_load[1] = _mm_load_pd(p + 2);
                sse_load[2] = _mm_load_pd(p + 4);
                sse_load[3] = _mm_load_pd(p + 6);
                sse_load2[0] = _mm_load_pd(p2);
                sse_load2[1] = _mm_load_pd(p2 + 2);
                sse_load2[2] = _mm_load_pd(p2 + 4);
                sse_load2[3] = _mm_load_pd(p2 + 6);
                __m128d temp = _mm_mul_pd(sse_load[0], sse_load2[0]);
                __m128d temp1 = _mm_mul_pd(sse_load[1], sse_load2[1]);
                __m128d temp2 = _mm_mul_pd(sse_load[2], sse_load2[2]);
                __m128d temp3 = _mm_mul_pd(sse_load[3], sse_load2[3]);
                const double * q = (const double*)&temp;
                const double * q1 = (const double*)&temp1;
                const double * q2 = (const double*)&temp2;
                const double * q3 = (const double*)&temp3;
                sum += q[0] + q[1];
                sum += q1[0] + q1[1];
                sum += q2[0] + q2[1];
                sum += q3[0] + q3[1];
                p += nBlockWidth;
                p2 += nBlockWidth;
        }

        for (size_t i = 0; i < reserved; i++)
        {
                sum += p[i] * p2[i];
        }
        return sum;

}


int double_mul_test(int argc, char ** argv)
{
        int n = 40000;
        double * a = (double*)TAlignAlloc(sizeof(double)*n, 16);
        for (int i = 0; i < n; i++)
        {
                a[i] = (double)(i % 10);
        }

        double * b = (double*)TAlignAlloc(sizeof(double)*n, 16);
        for (int i = 0; i < n; i++)
        {
                b[i] = (double)(i % 10);
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
        return 0;
}