/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
/** Precision-independent memory-related routines.
    (Shared by [sdcz]memory.c) **/

#include "slu_ddefs.h"


#if ( DEBUGlevel>=1 )           /* Debug malloc/free. */
int superlu_malloc_total = 0;

#define PAD_FACTOR  2
#define DWORD  (sizeof(double)) /* Be sure it's no smaller than double. */
/* size_t is usually defined as 'unsigned long' */

void *superlu_malloc(size_t size)
{
    char *buf;

    buf = (char *) malloc(size + DWORD);
    if ( !buf ) {
	printf("superlu_malloc fails: malloc_total %.0f MB, size %ld\n",
	       superlu_malloc_total*1e-6, size);
	ABORT("superlu_malloc: out of memory");
    }

    ((int_t *) buf)[0] = size;
#if 0
    superlu_malloc_total += size + DWORD;
#else
    superlu_malloc_total += size;
#endif
    return (void *) (buf + DWORD);
}

void superlu_free(void *addr)
{
    char *p = ((char *) addr) - DWORD;

    if ( !addr )
	ABORT("superlu_free: tried to free NULL pointer");

    if ( !p )
	ABORT("superlu_free: tried to free NULL+DWORD pointer");

    { 
	int_t n = ((int_t *) p)[0];
	
	if ( !n )
	    ABORT("superlu_free: tried to free a freed pointer");
	*((int_t *) p) = 0; /* Set to zero to detect duplicate free's. */
#if 0	
	superlu_malloc_total -= (n + DWORD);
#else
	superlu_malloc_total -= n;
#endif

	if ( superlu_malloc_total < 0 )
	    ABORT("superlu_malloc_total went negative!");
	
	/*free (addr);*/
	free (p);
    }

}

#else   /* production mode */

void *superlu_malloc(size_t size)
{
    void *buf;
    buf = (void *) malloc(size);
    return (buf);
}

void superlu_free(void *addr)
{
    free (addr);
}

#endif


/*! \brief Set up pointers for integer working arrays.
 */
void
SetIWork(int m, int n, int panel_size, int *iworkptr, int **segrep,
	 int **parent, int **xplore, int **repfnz, int **panel_lsub,
	 int **xprune, int **marker)
{
    *segrep = iworkptr;
    *parent = iworkptr + m;
    *xplore = *parent + m;
    *repfnz = *xplore + m;
    *panel_lsub = *repfnz + panel_size * m;
    *xprune = *panel_lsub + panel_size * m;
    *marker = *xprune + n;
    ifill (*repfnz, m * panel_size, EMPTY);
    ifill (*panel_lsub, m * panel_size, EMPTY);
}


void
copy_mem_int(int howmany, void *old, void *new)
{
    register int i;
    int *iold = old;
    int *inew = new;
    for (i = 0; i < howmany; i++) inew[i] = iold[i];
}


void
user_bcopy(char *src, char *dest, int bytes)
{
    char *s_ptr, *d_ptr;

    s_ptr = src + bytes - 1;
    d_ptr = dest + bytes - 1;
    for (; d_ptr >= dest; --s_ptr, --d_ptr ) *d_ptr = *s_ptr;
}



int *intMalloc(int n)
{
    int *buf;
    buf = (int *) SUPERLU_MALLOC((size_t) n * sizeof(int));
    if ( !buf ) {
	ABORT("SUPERLU_MALLOC fails for buf in intMalloc()");
    }
    return (buf);
}

int *intCalloc(int n)
{
    int *buf;
    register int i;
    buf = (int *) SUPERLU_MALLOC(n * sizeof(int));
    if ( !buf ) {
	ABORT("SUPERLU_MALLOC fails for buf in intCalloc()");
    }
    for (i = 0; i < n; ++i) buf[i] = 0;
    return (buf);
}



#if 0
check_expanders()
{
    int p;
    printf("Check expanders:\n");
    for (p = 0; p < NO_MEMTYPE; p++) {
	printf("type %d, size %d, mem %d\n",
	       p, expanders[p].size, (int)expanders[p].mem);
    }

    return 0;
}


StackInfo()
{
    printf("Stack: size %d, used %d, top1 %d, top2 %d\n",
	   stack.size, stack.used, stack.top1, stack.top2);
    return 0;
}



PrintStack(char *msg, GlobalLU_t *Glu)
{
    int i;
    int *xlsub, *lsub, *xusub, *usub;

    xlsub = Glu->xlsub;
    lsub  = Glu->lsub;
    xusub = Glu->xusub;
    usub  = Glu->usub;

    printf("%s\n", msg);
    
/*    printf("\nUCOL: ");
    for (i = 0; i < xusub[ndim]; ++i)
	printf("%f  ", ucol[i]);

    printf("\nLSUB: ");
    for (i = 0; i < xlsub[ndim]; ++i)
	printf("%d  ", lsub[i]);

    printf("\nUSUB: ");
    for (i = 0; i < xusub[ndim]; ++i)
	printf("%d  ", usub[i]);

    printf("\n");*/
    return 0;
}   
#endif



