/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 15, 2025.
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
/*
 * This needs to be in a separate file because zlib.h conflicts
 * with opflags.h.
 */
#include "compiler.h"
#include "zlib.h"
#include "macros.h"
#include "nasmlib.h"
#include "error.h"

/*
 * read line from standard macros set,
 * if there no more left -- return NULL
 */
static void *nasm_z_alloc(void *opaque, unsigned int items, unsigned int size)
{
    (void)opaque;
    return nasm_calloc(items, size);
}

static void nasm_z_free(void *opaque, void *ptr)
{
    (void)opaque;
    nasm_free(ptr);
}

char *uncompress_stdmac(const macros_t *sm)
{
    z_stream zs;
    void *buf = nasm_malloc(sm->dsize);

    nasm_zero(zs);
    zs.next_in   = (void *)sm->zdata;
    zs.avail_in  = sm->zsize;
    zs.next_out  = buf;
    zs.avail_out = sm->dsize;
    zs.zalloc    = nasm_z_alloc;
    zs.zfree     = nasm_z_free;

    if (inflateInit2(&zs, 0) != Z_OK)
        panic();

    if (inflate(&zs, Z_FINISH) != Z_STREAM_END)
        panic();

    inflateEnd(&zs);
    return buf;
}
