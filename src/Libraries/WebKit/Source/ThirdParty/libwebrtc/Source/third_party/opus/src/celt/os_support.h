/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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
#ifndef OS_SUPPORT_H
#define OS_SUPPORT_H

#ifdef CUSTOM_SUPPORT
#  include "custom_support.h"
#endif

#include "opus_types.h"
#include "opus_defines.h"

#include <string.h>
#include <stdlib.h>

/** Opus wrapper for malloc(). To do your own dynamic allocation, all you need to do is replace this function and opus_free */
#ifndef OVERRIDE_OPUS_ALLOC
static OPUS_INLINE void *opus_alloc (size_t size)
{
   return malloc(size);
}
#endif

/** Same as celt_alloc(), except that the area is only needed inside a CELT call (might cause problem with wideband though) */
#ifndef OVERRIDE_OPUS_ALLOC_SCRATCH
static OPUS_INLINE void *opus_alloc_scratch (size_t size)
{
   /* Scratch space doesn't need to be cleared */
   return opus_alloc(size);
}
#endif

/** Opus wrapper for free(). To do your own dynamic allocation, all you need to do is replace this function and opus_alloc */
#ifndef OVERRIDE_OPUS_FREE
static OPUS_INLINE void opus_free (void *ptr)
{
   free(ptr);
}
#endif

/** Copy n elements from src to dst. The 0* term provides compile-time type checking  */
#ifndef OVERRIDE_OPUS_COPY
#define OPUS_COPY(dst, src, n) (memcpy((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#endif

/** Copy n elements from src to dst, allowing overlapping regions. The 0* term
    provides compile-time type checking */
#ifndef OVERRIDE_OPUS_MOVE
#define OPUS_MOVE(dst, src, n) (memmove((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))
#endif

/** Set n elements of dst to zero */
#ifndef OVERRIDE_OPUS_CLEAR
#define OPUS_CLEAR(dst, n) (memset((dst), 0, (n)*sizeof(*(dst))))
#endif

/*#ifdef __GNUC__
#pragma GCC poison printf sprintf
#pragma GCC poison malloc free realloc calloc
#endif*/

#endif /* OS_SUPPORT_H */

