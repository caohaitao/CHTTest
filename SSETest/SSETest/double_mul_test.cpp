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
        const double *p = a;
        const double *p2 = b;

        __m128d temp;
        __m128d temp1;
        __m128d temp2;
        __m128d temp3;
        double * q;
        double * q1;
        double * q2;
        double * q3;

        int nBlockWidth = 2 * 4;
        size_t block = n / nBlockWidth;
        size_t reserved = n % nBlockWidth;

        for (size_t i = 0; i < block; ++i)
        {
                temp = _mm_mul_pd(_mm_load_pd(p), _mm_load_pd(p2));
                temp1 = _mm_mul_pd(_mm_load_pd(p + 2), _mm_load_pd(p2 + 2));
                temp2 = _mm_mul_pd(_mm_load_pd(p + 4), _mm_load_pd(p2 + 4));
                temp3 = _mm_mul_pd(_mm_load_pd(p + 6), _mm_load_pd(p2 + 6));
                q  = (double*)&temp;
                q1 = (double*)&temp1;
                q2 = (double*)&temp2;
                q3 = (double*)&temp3;
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

static double sse_mul_256(double * a, double * b, int n)
{
        double sum = 0;
        __m256d sse_load;
        __m256d sse_load2;
        const double *p = a;
        const double *p2 = b;

        int nBlockWidth = 4;
        size_t block = n / nBlockWidth;
        size_t reserved = n % nBlockWidth;

        for (size_t i = 0; i < block; ++i)
        {
                sse_load = _mm256_load_pd(p);
                sse_load2 = _mm256_load_pd(p2);
                __m256d temp = _mm256_mul_pd(sse_load, sse_load2);
                const double * q = (const double*)&temp;
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

static double sse_mul_256_loop4(double * a, double * b, int n)
{
        double sum = 0;
        const double *p = a;
        const double *p2 = b;

        __m256d temp;
        __m256d temp1;
        __m256d temp2;
        __m256d temp3;
        double * q;
        double * q1;
        double * q2;
        double * q3;

        int nBlockWidth = 4 * 4;
        size_t block = n / nBlockWidth;
        size_t reserved = n % nBlockWidth;

        for (size_t i = 0; i < block; ++i)
        {
                temp = _mm256_mul_pd(_mm256_load_pd(p), _mm256_load_pd(p2));
                temp1 = _mm256_mul_pd(_mm256_load_pd(p + 4), _mm256_load_pd(p2 + 4));
                temp2 = _mm256_mul_pd(_mm256_load_pd(p + 8), _mm256_load_pd(p2 + 8));
                temp3 = _mm256_mul_pd(_mm256_load_pd(p + 12), _mm256_load_pd(p2 + 12));
                q = (double*)&temp;
                q1 = (double*)&temp1;
                q2 = (double*)&temp2;
                q3 = (double*)&temp3;
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


int double_mul_test(int argc, char ** argv)
{
        int n = 40000;
        printf("n=%d\n", n);
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

        double sum = 0;

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
                sum = sse_mul_256(a, b, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("sse_mul_256,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = sse_mul_256_loop4(a, b, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("sse_mul_256_loop4,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }
        return 0;
}