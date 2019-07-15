#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include <chrono>
#include "nmmintrin.h"
#include "common.h"

static int normal_mul(int * a, int * b,int n)
{
        int sum = 0;
        for (int i = 0; i < n; i++)
        {
                sum += a[i] * b[i];
        }
        return sum;
}

static int normal_mul_loop4(int * a, int * b, int n)
{
        int sum = 0;
        size_t block = n / 4;
        size_t reserve = n % 4;
        int * p = a;
        int * p2 = b;
        for (size_t i = 0; i < block; ++i)
        {
                sum += *(p + 0)**(p2 + 0);
                sum += *(p + 1)**(p2 + 1);
                sum += *(p + 2)**(p2 + 2);
                sum += *(p + 3)**(p2 + 3);
                p += 4;
                p2 += 4;
        }

        for (size_t i = 0; i < reserve; ++i)
        {
                sum += p[i] * p2[i];
        }
        return sum;
}

static int sse_mul(int * a, int * b, int n)
{
        int sum = 0;
        __m128i sse_load;
        __m128i sse_load2;
        __m128i *p = (__m128i*)a;
        __m128i *p2 = (__m128i*)b;

        size_t block = n / 4;
        size_t reserve = n % 4;

        for (size_t i = 0; i < block; ++i)
        {
                sse_load = _mm_load_si128(p);
                sse_load2 = _mm_load_si128(p2);
                __m128i temp = _mm_mullo_epi32(sse_load, sse_load2);
                const int* q = (const int*)&temp;
                sum += q[0] + q[1] + q[2] + q[3];
                ++p;
                ++p2;
        }

        int * q = (int*)p;
        int * q2 = (int*)p2;

        for (size_t i = 0; i < reserve; ++i)
        {
                sum += q[i] * q2[i];
        }
        return sum;
}

int int_mul_test(int argc, char ** argv)
{
        int n = 40000;
        int * a = (int*)TAlignAlloc(sizeof(int)*n, 16);
        for (int i = 0; i < n; i++)
        {
                a[i] = i%10;
        }

        int * b = (int*)TAlignAlloc(sizeof(int)*n, 16);
        for (int i = 0; i < n; i++)
        {
                b[i] = i;
        }

        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;//∫¡√Î
        typedef std::chrono::duration<float> fsec;

        int sum = 0;

        {
                auto t0 = Time::now();
                sum = normal_mul(a, b,n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("normal_num,value(%d) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = normal_mul_loop4(a, b, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("normal_mul_loop4,value(%d) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = sse_mul(a, b, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("sse_mul,value(%d) use_time(%0.4f ms)\n", sum, d.count());
        }
        return 0;
}