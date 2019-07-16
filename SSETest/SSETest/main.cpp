#include "stdio.h"

extern int int_add_test(int argc, char ** argv);
extern int float_add_test(int argc, char ** argv);
extern int int_mul_test(int argc, char ** argv);
extern int float_mul_test(int argc, char ** argv);
extern int double_add_test(int argc, char ** argv);
extern int double_mul_test(int argc, char ** argv);
extern int sqrt_test(int argc, char ** argv);

int main(int argc, char ** argv)
{
        //printf("------------------int add test----------\n");
        //int_add_test(argc, argv);
        //printf("------------------float add test--------\n");
        //float_add_test(argc, argv);
        //printf("------------------double add test-------\n");
        //double_add_test(argc, argv);
        //printf("------------------int mul test----------\n");
        //int_mul_test(argc, argv);
        //printf("------------------float mul test--------\n");
        //float_mul_test(argc, argv);
        //printf("------------------double mul test-------\n");
        //double_mul_test(argc, argv);
        printf("------------------sqrt test-------------\n");
        sqrt_test(argc, argv);
        return 0;
}