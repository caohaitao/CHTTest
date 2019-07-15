#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include <chrono>
#include "nmmintrin.h"
#include <immintrin.h>
#include "common.h"

static float normal_add(float * a, int n)
{
        float sum = 0;
        for (int i = 0; i < n; i++)
        {
                sum += a[i];
        }
        return sum;
}

static float nomral_add_loop4(float * a, int n)
{
        float sum = 0;
        size_t block = n / 4;    // 等价于n >> 2
        size_t reserve = n % 4;  // 等价于 n & 0x3
        float *p = a;

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

static float sse_add(float *a, size_t n) {
        float s = 0;
        int cntBlock = n / 4;
        int cntRem = n%4;
        __m128 fSum = _mm_setzero_ps();//求和变量，初值清零
        __m128 fLoad;
        const float*p = a;

        for (unsigned int i = 0; i<cntBlock; i++)
        {
                fLoad = _mm_load_ps(p);//加载
                fSum = _mm_add_ps(fSum, fLoad);//求和

                p += 4;
        }
        const float *q = (const float*)&fSum;
        s = q[0] + q[1] + q[2] + q[3];        //合并

        for (int i = 0; i<cntRem; i++)//处理尾部剩余数据
        {
                s += p[i];
        }
        return s;
}

static float sse_add_256(float *a, size_t n)
{
        float s = 0;
        int cntBlock = n / 8;
        int cntRem = n % 8;
        __m256 fSum = _mm256_setzero_ps();//求和变量，初值清零
        __m256 fLoad;
        const float*p = a;

        for (unsigned int i = 0; i<cntBlock; i++)
        {
                fLoad = _mm256_load_ps(p);//加载
                fSum = _mm256_add_ps(fSum, fLoad);//求和

                p += 8;
        }
        const float *q = (const float*)&fSum;
        s = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];        //合并

        for (int i = 0; i<cntRem; i++)//处理尾部剩余数据
        {
                s += p[i];
        }
        return s;
}

static float sse_add_loop4(float *a, int n) {
        float s = 0;
        unsigned int nBlockWidth = 4 * 4;
        unsigned int cntBlock = n / nBlockWidth;
        unsigned int cntRem = n%nBlockWidth;
        __m128 fSum0 = _mm_setzero_ps();//求和变量，初值清零
        __m128 fSum1 = _mm_setzero_ps();
        __m128 fSum2 = _mm_setzero_ps();
        __m128 fSum3 = _mm_setzero_ps();
        __m128 fLoad0, fLoad1, fLoad2, fLoad3;
        const float *p = a;
        for (unsigned int i = 0; i<cntBlock; i++)
        {
                fLoad0 = _mm_load_ps(p);//加载
                fLoad1 = _mm_load_ps(p + 4);
                fLoad2 = _mm_load_ps(p + 8);
                fLoad3 = _mm_load_ps(p + 12);
                fSum0 = _mm_add_ps(fSum0, fLoad0);//求和
                fSum1 = _mm_add_ps(fSum1, fLoad1);
                fSum2 = _mm_add_ps(fSum2, fLoad2);
                fSum3 = _mm_add_ps(fSum3, fLoad3);
                p += nBlockWidth;
        }
        fSum0 = _mm_add_ps(fSum0, fSum1);
        fSum2 = _mm_add_ps(fSum2, fSum3);
        fSum0 = _mm_add_ps(fSum0, fSum2);
        const float*q = (const float*)&fSum0;
        s = q[0] + q[1] + q[2] + q[3];                //合并
        for (unsigned int i = 0; i<cntRem; i++)//处理尾部剩余数据
        {
                s += p[i];
        }
        return s;
}

static float sse_add_256_loop4(float *a, int n) {
        float s = 0;
        unsigned int nBlockWidth = 8 * 4;
        unsigned int cntBlock = n / nBlockWidth;
        unsigned int cntRem = n%nBlockWidth;
        __m256 fSum0 = _mm256_setzero_ps();//求和变量，初值清零
        __m256 fSum1 = _mm256_setzero_ps();
        __m256 fSum2 = _mm256_setzero_ps();
        __m256 fSum3 = _mm256_setzero_ps();
        __m256 fLoad0, fLoad1, fLoad2, fLoad3;
        const float *p = a;
        for (unsigned int i = 0; i<cntBlock; i++)
        {
                fLoad0 = _mm256_load_ps(p);//加载
                fLoad1 = _mm256_load_ps(p + 8);
                fLoad2 = _mm256_load_ps(p + 16);
                fLoad3 = _mm256_load_ps(p + 24);
                fSum0 = _mm256_add_ps(fSum0, fLoad0);//求和
                fSum1 = _mm256_add_ps(fSum1, fLoad1);
                fSum2 = _mm256_add_ps(fSum2, fLoad2);
                fSum3 = _mm256_add_ps(fSum3, fLoad3);
                p += nBlockWidth;
        }
        fSum0 = _mm256_add_ps(fSum0, fSum1);
        fSum2 = _mm256_add_ps(fSum2, fSum3);
        fSum0 = _mm256_add_ps(fSum0, fSum2);
        const float*q = (const float*)&fSum0;
        s = q[0] + q[1] + q[2] + q[3]+ q[4] + q[5] + q[6] + q[7];                //合并
        for (unsigned int i = 0; i<cntRem; i++)//处理尾部剩余数据
        {
                s += p[i];
        }
        return s;
}

int float_add_test(int argc, char ** argv)
{
        srand(time(0));
        int n = 40000;
        printf("n=%d\n", n);
        float * a = (float*)TAlignAlloc(sizeof(float)*n, 16);
        for (int i = 0; i < n; i++)
        {
                a[i] = (float)i;
        }

        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;//毫秒
        typedef std::chrono::duration<float> fsec;

        float sum = 0;

        {
                auto t0 = Time::now();
                sum = normal_add(a, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("normal add,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }


        {
                auto t0 = Time::now();
                sum = nomral_add_loop4(a, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("nomral_add_loop4,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = sse_add(a, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("sse_add,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = sse_add_256(a, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("sse_add_256,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = sse_add_loop4(a, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("sse_add_loop4,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }

        {
                auto t0 = Time::now();
                sum = sse_add_256_loop4(a, n);
                auto t1 = Time::now();
                fsec fs = t1 - t0;
                ms d = std::chrono::duration_cast<ms>(fs);

                printf("sse_add_256_loop4,value(%0.4f) use_time(%0.4f ms)\n", sum, d.count());
        }

        return 0;
}