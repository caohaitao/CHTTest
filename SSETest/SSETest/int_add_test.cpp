#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include <chrono>
#include "nmmintrin.h"
#include "common.h"

static int normal_add(int * a, int n)
{
        int sum = 0;
        for (int i = 0; i < n; i++)
        {
                sum += a[i];
        }
        return sum;
}

static int nomral_add_loop4(int * a, int n)
{
        int sum = 0;
        size_t block = n / 4;    // 等价于n >> 2
        size_t reserve = n % 4;  // 等价于 n & 0x3
        int *p = a;

        for (size_t i = 0; i < block; ++i) {
                sum += *p;
                sum += *(p + 1);
                sum += *(p + 2);
                sum += *(p + 3);
                p += 4;
        }

        // 剩余的不足4字节
        for (size_t i = 0; i < reserve; ++i) {
                sum += p[i];
        }
        return sum;
}

static int sse_add(int *a, size_t n) {
        int sum = 0;
        __m128i sse_sum = _mm_setzero_si128();
        __m128i sse_load;
        __m128i *p = (__m128i*)a;

        size_t block = n / 4;     // SSE寄存器能一次处理4个32位的整数
        size_t reserve = n % 4;  // 剩余的不足16字节

        for (size_t i = 0; i < block; ++i) {
                sse_load = _mm_load_si128(p);
                sse_sum = _mm_add_epi32(sse_sum, sse_load); // 带符号32位紧缩加法
                ++p;
        }

        // 剩余的不足16字节
        int *q = (int *)p;
        for (size_t i = 0; i < reserve; ++i) {
                sum += q[i];
        }

        // 将累加值合并
        sse_sum = _mm_hadd_epi32(sse_sum, sse_sum);  // 带符号32位水平加法
        sse_sum = _mm_hadd_epi32(sse_sum, sse_sum);

        sum += _mm_cvtsi128_si32(sse_sum);  // 返回低32位
        return sum;
}

static int sse_add_loop4(int *a, int n) {
        int sum = 0;
        size_t block = n / 16;    // SSE寄存器能一次处理4个32位的整数
        size_t reserve = n % 16; // 剩余的字节

        __m128i sse_sum0 = _mm_setzero_si128();
        __m128i sse_sum1 = _mm_setzero_si128();
        __m128i sse_sum2 = _mm_setzero_si128();
        __m128i sse_sum3 = _mm_setzero_si128();
        __m128i sse_load0;
        __m128i sse_load1;
        __m128i sse_load2;
        __m128i sse_load3;
        __m128i *p = (__m128i*)a;

        for (size_t i = 0; i < block; ++i) {
                sse_load0 = _mm_load_si128(p);
                sse_load1 = _mm_load_si128(p + 1);
                sse_load2 = _mm_load_si128(p + 2);
                sse_load3 = _mm_load_si128(p + 3);

                sse_sum0 = _mm_add_epi32(sse_sum0, sse_load0);
                sse_sum1 = _mm_add_epi32(sse_sum1, sse_load1);
                sse_sum2 = _mm_add_epi32(sse_sum2, sse_load2);
                sse_sum3 = _mm_add_epi32(sse_sum3, sse_load3);

                p += 4;
        }

        // 剩余的不足16字节
        int *q = (int *)p;
        for (size_t i = 0; i < reserve; ++i) {
                sum += q[i];
        }

        // 将累加值两两合并
        sse_sum0 = _mm_add_epi32(sse_sum0, sse_sum1);
        sse_sum2 = _mm_hadd_epi32(sse_sum2, sse_sum3);
        sse_sum0 = _mm_add_epi32(sse_sum0, sse_sum2);

        sse_sum0 = _mm_hadd_epi32(sse_sum0, sse_sum0);
        sse_sum0 = _mm_hadd_epi32(sse_sum0, sse_sum0);

        sum += _mm_cvtsi128_si32(sse_sum0); // 取低32位

        return sum;
}

int int_add_test(int argc, char ** argv)
{
        srand(time(0));
        int n = 40000;
        printf("n=%d\n", n);
        int * a = (int*)TAlignAlloc(sizeof(int)*n, 16);
        for (int i = 0; i < n; i++)
        {
                a[i] = i;
        }

        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;//毫秒
        typedef std::chrono::duration<float> fsec;

        int sum = 0;

        {
                auto t0 = Time::now();
                sum = normal_add(a, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("normal add,value(%d) use_time(%0.4f ms)\n", sum, d.count());
        }


        {
                auto t0 = Time::now();
                sum = nomral_add_loop4(a, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("nomral_add_loop4,value(%d) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = sse_add(a, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("sse_add,value(%d) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = sse_add_loop4(a, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("sse_add_loop4,value(%d) use_time(%0.4f ms)\n", sum, d.count());
        }

        return 0;
}