/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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
#pragma once

#if OS(DARWIN)

#if !PLATFORM(IOS_FAMILY_SIMULATOR) && __has_include(<libproc.h>)
#    include <libproc.h>
#    if RUSAGE_INFO_CURRENT >= 4
#        define HAS_MAX_FOOTPRINT
#        if defined(RLIMIT_FOOTPRINT_INTERVAL) && __has_include(<libproc_internal.h>) && PLATFORM(COCOA)
#            define HAS_RESET_FOOTPRINT_INTERVAL
#            define MAX_FOOTPRINT_FIELD ri_interval_max_phys_footprint
#            include <libproc_internal.h>
#        else
#            define MAX_FOOTPRINT_FIELD ri_lifetime_max_phys_footprint
#        endif
#    else
#        define HAS_ONLY_PHYS_FOOTPRINT
#    endif
#endif

struct ProcessMemoryFootprint {
public:
    uint64_t current;
    uint64_t peak;

    static ProcessMemoryFootprint now()
    {
#ifdef HAS_MAX_FOOTPRINT
        rusage_info_v4 rusage;
        if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&rusage))
            return { 0L, 0L };

        return { rusage.ri_phys_footprint, rusage.MAX_FOOTPRINT_FIELD };
#elif defined(HAS_ONLY_PHYS_FOOTPRINT)
        rusage_info_v0 rusage;
        if (proc_pid_rusage(getpid(), RUSAGE_INFO_V0, (rusage_info_t *)&rusage))
            return { 0L, 0L };

        return { rusage.ri_phys_footprint, 0L };
#else
        return { 0L, 0L };
#endif
    }

    static void resetPeak()
    {
#ifdef HAS_RESET_FOOTPRINT_INTERVAL
        proc_reset_footprint_interval(getpid());
#endif
    }
};

#endif
