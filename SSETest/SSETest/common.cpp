#include "common.h"
#include "stdlib.h"

void * TAlignAlloc(size_t alloc_size, int alignment)
{
        int asize = (alloc_size + alignment - 1)&(-alignment);
        return _aligned_malloc(asize, alignment);
}

void TAlignFree(void * p)
{
        _aligned_free(p);
}
