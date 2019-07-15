#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include <chrono>
#include "nmmintrin.h"
#include <immintrin.h>
#include "common.h"

static double normal_add(double * a, int n)
{
        double sum = 0;
        for (size_t i = 0; i < n; i++)
        {
                sum += a[i];
        }
        return sum;
}

static double nomral_add_loop4(double * a, int n)
{
        double sum = 0;
        size_t block = n / 4;    // 等价于n >> 2
        size_t reserve = n % 4;  // 等价于 n & 0x3
        double *p = a;

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

static double sse_add(double * a, size_t n)
{
        double s = 0;
        int cntBlock = n / 2;
        int cntRem = n % 2;
        __m128d fSum = _mm_setzero_pd();//求和变量，初值清零
        __m128d fLoad;
        const double*p = a;

        for (unsigned int i = 0; i<cntBlock; i++)
        {
                fLoad = _mm_load_pd(p);
                fSum = _mm_add_pd(fSum, fLoad);//求和

                p += 2;
        }
        const double *q = (const double*)&fSum;
        s = q[0] + q[1];        //合并

        for (int i = 0; i<cntRem; i++)//处理尾部剩余数据
        {
                s += p[i];
        }
        return s;
}

static double sse_add_256(double * a, size_t n)
{
        double s = 0;
        int cntBlock = n / 4;
        int cntRem = n % 4;
        __m256d fSum = _mm256_setzero_pd();//求和变量，初值清零
        __m256d fLoad;
        const double*p = a;

        for (unsigned int i = 0; i<cntBlock; i++)
        {
                fLoad = _mm256_load_pd(p);
                fSum = _mm256_add_pd(fSum, fLoad);//求和

                p += 4;
        }
        const double *q = (const double*)&fSum;
        s = q[0] + q[1] + q[2] + q[3];        //合并

        for (int i = 0; i<cntRem; i++)//处理尾部剩余数据
        {
                s += p[i];
        }
        return s;
}

static double sse_add_loop4(double *a, int n) {
        double s = 0;
        unsigned int nBlockWidth = 2 * 4;
        unsigned int cntBlock = n / nBlockWidth;
        unsigned int cntRem = n%nBlockWidth;
        __m128d fSum0 = _mm_setzero_pd();//求和变量，初值清零
        __m128d fSum1 = _mm_setzero_pd();
        __m128d fSum2 = _mm_setzero_pd();
        __m128d fSum3 = _mm_setzero_pd();
        __m128d fLoad0, fLoad1, fLoad2, fLoad3;
        const double *p = a;
        for (unsigned int i = 0; i<cntBlock; i++)
        {
                fLoad0 = _mm_load_pd(p);//加载
                fLoad1 = _mm_load_pd(p + 2);
                fLoad2 = _mm_load_pd(p + 4);
                fLoad3 = _mm_load_pd(p + 6);
                fSum0 = _mm_add_pd(fSum0, fLoad0);//求和
                fSum1 = _mm_add_pd(fSum1, fLoad1);
                fSum2 = _mm_add_pd(fSum2, fLoad2);
                fSum3 = _mm_add_pd(fSum3, fLoad3);
                p += nBlockWidth;
        }
        fSum0 = _mm_add_pd(fSum0, fSum1);
        fSum2 = _mm_add_pd(fSum2, fSum3);
        fSum0 = _mm_add_pd(fSum0, fSum2);
        const double*q = (const double*)&fSum0;
        s = q[0] + q[1];                //合并
        for (unsigned int i = 0; i<cntRem; i++)//处理尾部剩余数据
        {
                s += p[i];
        }
        return s;
}

static double sse_add_256_loop4(double *a, int n) {
        double s = 0;
        unsigned int nBlockWidth = 4 * 4;
        unsigned int cntBlock = n / nBlockWidth;
        unsigned int cntRem = n%nBlockWidth;
        __m256d fSum0 = _mm256_setzero_pd();//求和变量，初值清零
        __m256d fSum1 = _mm256_setzero_pd();
        __m256d fSum2 = _mm256_setzero_pd();
        __m256d fSum3 = _mm256_setzero_pd();
        __m256d fLoad0, fLoad1, fLoad2, fLoad3;
        const double *p = a;
        for (unsigned int i = 0; i<cntBlock; i++)
        {
                fLoad0 = _mm256_load_pd(p);//加载
                fLoad1 = _mm256_load_pd(p + 4);
                fLoad2 = _mm256_load_pd(p + 8);
                fLoad3 = _mm256_load_pd(p + 12);
                fSum0 = _mm256_add_pd(fSum0, fLoad0);//求和
                fSum1 = _mm256_add_pd(fSum1, fLoad1);
                fSum2 = _mm256_add_pd(fSum2, fLoad2);
                fSum3 = _mm256_add_pd(fSum3, fLoad3);
                p += nBlockWidth;
        }
        fSum0 = _mm256_add_pd(fSum0, fSum1);
        fSum2 = _mm256_add_pd(fSum2, fSum3);
        fSum0 = _mm256_add_pd(fSum0, fSum2);
        const double*q = (const double*)&fSum0;
        s = q[0] + q[1] + q[2] + q[3];                //合并
        for (unsigned int i = 0; i<cntRem; i++)//处理尾部剩余数据
        {
                s += p[i];
        }
        return s;
}

int double_add_test(int argc, char ** argv)
{
        srand(time(0));
        int n = 40000;
        printf("n=%d\n", n);
        double * a = (double*)TAlignAlloc(sizeof(double)*n, 16);
        for (int i = 0; i < n; i++)
        {
                a[i] = (double)i;
        }

        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;//毫秒
        typedef std::chrono::duration<float> fsec;

        double sum = 0;

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