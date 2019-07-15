#include "stdio.h"

extern int int_add_test(int argc, char ** argv);
extern int float_add_test(int argc, char ** argv);
extern int int_mul_test(int argc, char ** argv);
extern int float_mul_test(int argc, char ** argv);
extern int double_add_test(int argc, char ** argv);
extern int double_mul_test(int argc, char ** argv);

int main(int argc, char ** argv)
{
        //printf("------------------float-------------\n");
        //float_add_test(argc, argv);
        //printf("------------------double-------------\n");
        //double_add_test(argc, argv);

        //float_mul_test(argc, argv);
        double_mul_test(argc, argv);
        return 0;
}