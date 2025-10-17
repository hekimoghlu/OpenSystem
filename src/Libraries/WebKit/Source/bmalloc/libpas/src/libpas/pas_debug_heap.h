/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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
#ifndef PAS_DEBUG_HEAP_H
#define PAS_DEBUG_HEAP_H

#include "pas_allocation_mode.h"
#include "pas_allocation_result.h"
#include "pas_heap_config_kind.h"
#include "pas_log.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

/* Bmalloc has a DebugHeap singleton that can be used to divert bmalloc calls to system malloc.
   When libpas is used in bmalloc, we use this to glue libpas into that mechanism. */

#if PAS_BMALLOC

// FIXME: Find a way to declare bmalloc's symbol visibility without having to
// import a bmalloc header.
#include "BExport.h"

/* The implementations are provided by bmalloc. */
BEXPORT extern bool pas_debug_heap_is_enabled(pas_heap_config_kind);
BEXPORT extern void* pas_debug_heap_malloc(size_t);
BEXPORT extern void* pas_debug_heap_memalign(size_t alignment, size_t);
BEXPORT extern void* pas_debug_heap_realloc(void* ptr, size_t);
BEXPORT extern void pas_debug_heap_free(void* ptr);

#else /* PAS_BMALLOC -> so !PAS_BMALLOC */

static inline bool pas_debug_heap_is_enabled(pas_heap_config_kind kind)
{
    PAS_UNUSED_PARAM(kind);
    return false;
}

static inline void* pas_debug_heap_malloc(size_t size)
{
    PAS_UNUSED_PARAM(size);
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

static inline void* pas_debug_heap_memalign(size_t alignment, size_t size)
{
    PAS_UNUSED_PARAM(alignment);
    PAS_UNUSED_PARAM(size);
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

static inline void* pas_debug_heap_realloc(void* ptr, size_t size)
{
    PAS_UNUSED_PARAM(ptr);
    PAS_UNUSED_PARAM(size);
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

static inline void pas_debug_heap_free(void* ptr)
{
    PAS_UNUSED_PARAM(ptr);
    PAS_ASSERT(!"Should not be reached");
}

#endif /* PAS_BMALLOC -> so end of !PAS_BMALLOC */

static inline pas_allocation_result pas_debug_heap_allocate(size_t size, size_t alignment, pas_allocation_mode allocation_mode)
{
    static const bool verbose = false;
    
    pas_allocation_result result;
    void* raw_result;
    
    if (alignment > sizeof(void*)) {
        if (verbose)
            pas_log("Going down debug memalign path.\n");
        raw_result = pas_debug_heap_memalign(alignment, size);
    } else {
        if (verbose)
            pas_log("Going down debug malloc path.\n");
        raw_result = pas_debug_heap_malloc(size);
    }

    if (verbose)
        pas_log("raw_result = %p\n", raw_result);

    result.did_succeed = !!raw_result;
    result.begin = (uintptr_t)raw_result;
    result.zero_mode = pas_zero_mode_may_have_non_zero;
    PAS_PROFILE(DEBUG_HEAP_ALLOCATION, result.begin, size, allocation_mode);

    return result;
}

PAS_END_EXTERN_C;

#endif /* PAS_DEBUG_HEAP_H */
