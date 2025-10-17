/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "bmalloc_heap_config.h"
#include "bmalloc_heap_inlines.h"
#include "pas_darwin_spi.h"
#include "pas_malloc_stack_logging.h"
#include <stdlib.h>

PAS_BEGIN_EXTERN_C;

pas_msl_is_enabled_flag pas_msl_is_enabled_flag_value = pas_msl_is_enabled_flag_indeterminate;
static void compute_msl_status(void)
{
    pas_msl_is_enabled_flag_value = getenv("MallocStackLogging") ? pas_msl_is_enabled_flag_enabled : pas_msl_is_enabled_flag_disabled;
}

bool pas_compute_msl_is_enabled(void)
{
    static pthread_once_t key = PTHREAD_ONCE_INIT;
    pthread_once(&key, compute_msl_status);
    return pas_msl_is_enabled_flag_value == pas_msl_is_enabled_flag_enabled;
}

#if PAS_OS(DARWIN)

PAS_NEVER_INLINE pas_allocation_result pas_msl_malloc_logging_slow(size_t size, pas_allocation_result result)
{
    PAS_TESTING_ASSERT(malloc_logger);
    if (result.did_succeed && pas_msl_is_enabled())
        malloc_logger(pas_stack_logging_type_alloc, (uintptr_t)0, (uintptr_t)size, 0, (uintptr_t)result.begin, 0);
    return result;
}

PAS_NEVER_INLINE void pas_msl_free_logging_slow(void* ptr)
{
    PAS_TESTING_ASSERT(malloc_logger);
    if (ptr && pas_msl_is_enabled())
        malloc_logger(pas_stack_logging_type_dealloc, (uintptr_t)0, (uintptr_t)ptr, 0, 0, 0);
}

#endif

PAS_END_EXTERN_C;

#endif /* LIBPAS_ENABLED */
