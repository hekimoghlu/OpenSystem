/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
#include "util.h"

#include "coretype.h"
#include "errwarn.h"


#ifdef WITH_DMALLOC
#undef yasm_xmalloc
#undef yasm_xcalloc
#undef yasm_xrealloc
#undef yasm_xfree
#endif

static /*@only@*/ /*@out@*/ void *def_xmalloc(size_t size);
static /*@only@*/ void *def_xcalloc(size_t nelem, size_t elsize);
static /*@only@*/ void *def_xrealloc
    (/*@only@*/ /*@out@*/ /*@returned@*/ /*@null@*/ void *oldmem, size_t size)
    /*@modifies oldmem@*/;
static void def_xfree(/*@only@*/ /*@out@*/ /*@null@*/ void *p)
    /*@modifies p@*/;

/* storage for global function pointers */
YASM_LIB_DECL
/*@only@*/ /*@out@*/ void * (*yasm_xmalloc) (size_t size) = def_xmalloc;
YASM_LIB_DECL
/*@only@*/ void * (*yasm_xcalloc) (size_t nelem, size_t elsize) = def_xcalloc;
YASM_LIB_DECL
/*@only@*/ void * (*yasm_xrealloc)
    (/*@only@*/ /*@out@*/ /*@returned@*/ /*@null@*/ void *oldmem, size_t size)
    /*@modifies oldmem@*/ = def_xrealloc;
YASM_LIB_DECL
void (*yasm_xfree) (/*@only@*/ /*@out@*/ /*@null@*/ void *p)
    /*@modifies p@*/ = def_xfree;


static void *
def_xmalloc(size_t size)
{
    void *newmem;

    if (size == 0)
        size = 1;
    newmem = malloc(size);
    if (!newmem)
        yasm__fatal(N_("out of memory"));

    return newmem;
}

static void *
def_xcalloc(size_t nelem, size_t elsize)
{
    void *newmem;

    if (nelem == 0 || elsize == 0)
        nelem = elsize = 1;

    newmem = calloc(nelem, elsize);
    if (!newmem)
        yasm__fatal(N_("out of memory"));

    return newmem;
}

static void *
def_xrealloc(void *oldmem, size_t size)
{
    void *newmem;

    if (size == 0)
        size = 1;
    if (!oldmem)
        newmem = malloc(size);
    else
        newmem = realloc(oldmem, size);
    if (!newmem)
        yasm__fatal(N_("out of memory"));

    return newmem;
}

static void
def_xfree(void *p)
{
    if (!p)
        return;
    free(p);
}
