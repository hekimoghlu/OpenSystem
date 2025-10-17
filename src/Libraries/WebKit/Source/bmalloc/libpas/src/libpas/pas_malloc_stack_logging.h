/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 15, 2023.
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
#ifndef PAS_MALLOC_STACK_LOGGING_H
#define PAS_MALLOC_STACK_LOGGING_H

#include "pas_allocation_result.h"
#include "pas_darwin_spi.h"
#include "pas_heap_config.h"
#include "pas_root.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_msl_is_enabled_flag {
    pas_msl_is_enabled_flag_enabled,
    pas_msl_is_enabled_flag_disabled,
    pas_msl_is_enabled_flag_indeterminate,
};
typedef enum pas_msl_is_enabled_flag pas_msl_is_enabled_flag;

extern pas_msl_is_enabled_flag pas_msl_is_enabled_flag_value;
PAS_API bool pas_compute_msl_is_enabled(void);

#if PAS_OS(DARWIN)
PAS_API PAS_NEVER_INLINE pas_allocation_result pas_msl_malloc_logging_slow(size_t size, pas_allocation_result result);
PAS_API PAS_NEVER_INLINE void pas_msl_free_logging_slow(void*);
#endif

static PAS_ALWAYS_INLINE bool pas_msl_is_enabled(void)
{
    switch (pas_msl_is_enabled_flag_value) {
    case pas_msl_is_enabled_flag_indeterminate:
        return pas_compute_msl_is_enabled();
    case pas_msl_is_enabled_flag_enabled:
        return true;
    case pas_msl_is_enabled_flag_disabled:
        return false;
    }
    return false;
}

static PAS_ALWAYS_INLINE pas_allocation_result pas_msl_malloc_logging(size_t size, pas_allocation_result result)
{
#if PAS_OS(DARWIN) && !defined(__swift__) // FIXME: Workaround for rdar://119319825
    if (PAS_UNLIKELY(malloc_logger))
        return pas_msl_malloc_logging_slow(size, result); /* Keep it tail-call to avoid messing up the fast path code. */
#else
    PAS_UNUSED_PARAM(size);
#endif
    return result;
}

static PAS_ALWAYS_INLINE void pas_msl_free_logging(void* ptr)
{
#if PAS_OS(DARWIN) && !defined(__swift__) // FIXME: Workaround for rdar://119319825

    if (PAS_UNLIKELY(malloc_logger))
        return pas_msl_free_logging_slow(ptr); /* Keep it tail-call to avoid messing up the fast path code. */
#else
    PAS_UNUSED_PARAM(ptr);
#endif
}

PAS_END_EXTERN_C;

#endif /* PAS_MALLOC_STACK_LOGGING_H */
