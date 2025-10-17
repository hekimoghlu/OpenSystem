/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 15, 2023.
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

#if OS(LINUX)

#include <sys/resource.h>
#include <wtf/linux/CurrentProcessMemoryStatus.h>

struct ProcessMemoryFootprint {
    uint64_t current;
    uint64_t peak;

    static ProcessMemoryFootprint now()
    {
        struct rusage ru;
        getrusage(RUSAGE_SELF, &ru);

        ProcessMemoryStatus ps;
        currentProcessMemoryStatus(ps);

        return { ps.resident, static_cast<uint64_t>(ru.ru_maxrss) * 1024 };
    }

    static void resetPeak()
    {
        // To reset the peak size, we need to write 5 to /proc/self/clear_refs
        // as described in `man -s5 proc`, in the clear_refs section.
        // Only available since 4.0.
        FILE* f = fopen("/proc/self/clear_refs", "w");
        if (!f)
            return;
        fwrite("5", 1, 1, f);
        fclose(f);
    }
};

#endif
