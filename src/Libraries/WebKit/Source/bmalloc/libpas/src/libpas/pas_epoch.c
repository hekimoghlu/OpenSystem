/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 5, 2025.
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

#include "pas_epoch.h"

#include "pas_log.h"
#include "pas_monotonic_time.h"

uint64_t pas_current_epoch;
bool pas_epoch_is_counter = false;

uint64_t pas_get_epoch(void)
{
    static const bool verbose = false;
    static bool first = true;
    
    uint64_t result;
    
    if (pas_epoch_is_counter) {
        /* This is just for testing. */
        result = ++pas_current_epoch;
    } else
        result = pas_get_current_monotonic_time_nanoseconds();

    PAS_ASSERT(result >= PAS_EPOCH_MIN);
    PAS_ASSERT(result <= PAS_EPOCH_MAX);

    if (first) {
        if (verbose)
            pas_log("first epoch = %llu\n", (unsigned long long)result);
        first = false;
    }

    return result;
}

#endif /* LIBPAS_ENABLED */
