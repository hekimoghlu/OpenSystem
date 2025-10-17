/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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

#include "pas_monotonic_time.h"
#if PAS_OS(DARWIN)
#include <mach/mach_time.h>
#endif

#if PAS_OS(DARWIN)
static mach_timebase_info_data_t timebase_info;
static mach_timebase_info_data_t* timebase_info_ptr;

static PAS_NEVER_INLINE mach_timebase_info_data_t* get_timebase_info_slow(void)
{
    kern_return_t kern_return;
    kern_return = mach_timebase_info(&timebase_info);
    PAS_ASSERT(kern_return == KERN_SUCCESS);
    pas_fence();
    timebase_info_ptr = &timebase_info;
    return &timebase_info;
}

static mach_timebase_info_data_t* get_timebase_info(void)
{
    mach_timebase_info_data_t* result;

    result = timebase_info_ptr;
    if (PAS_LIKELY(result))
        return result;

    return get_timebase_info_slow();
}

uint64_t pas_get_current_monotonic_time_nanoseconds(void)
{
    uint64_t result;
    mach_timebase_info_data_t* info;

    info = get_timebase_info();

    result = mach_approximate_time();
    result *= info->numer;
    result /= info->denom;

    return result;
}

#elif PAS_OS(LINUX)

uint64_t pas_get_current_monotonic_time_nanoseconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_COARSE, &ts);
    return ts.tv_sec * 1.0e9 + ts.tv_nsec;
}

#elif PAS_PLATFORM(PLAYSTATION)

uint64_t pas_get_current_monotonic_time_nanoseconds(void)
{
    struct timespec ts;
    clock_gettime_np(CLOCK_MONOTONIC_FAST, &ts);
    return ts.tv_sec * 1000u * 1000u * 1000u + ts.tv_nsec;
}

#endif

#endif /* LIBPAS_ENABLED */
