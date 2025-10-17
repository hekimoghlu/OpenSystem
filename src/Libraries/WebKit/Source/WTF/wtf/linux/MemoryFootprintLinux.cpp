/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 27, 2023.
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
#include "config.h"
#include <wtf/MemoryFootprint.h>

#include <stdio.h>
#include <wtf/MonotonicTime.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/StringView.h>

namespace WTF {

static const Seconds s_memoryFootprintUpdateInterval = 1_s;

template<typename Functor>
static void forEachLine(FILE* file, Functor functor)
{
    char* buffer = nullptr;
    size_t size = 0;
    while (getline(&buffer, &size, file) != -1) {
        functor(buffer);
    }
    free(buffer);
}

static size_t computeMemoryFootprint()
{
    FILE* file = fopen("/proc/self/smaps", "r");
    if (!file)
        return 0;

    unsigned long totalPrivateDirtyInKB = 0;
    bool isAnonymous = false;
    forEachLine(file, [&] (char* buffer) {
        {
            unsigned long start;
            unsigned long end;
            unsigned long offset;
            unsigned long inode;
            char dev[32];
            char perms[5];
            char path[7];
            int scannedCount = sscanf(buffer, "%lx-%lx %4s %lx %31s %lu %6s", &start, &end, perms, &offset, dev, &inode, path);
            if (scannedCount == 6) {
                isAnonymous = true;
                return;
            }
            if (scannedCount == 7) {
                auto pathString = StringView::fromLatin1(path);
                isAnonymous = pathString == "[heap]"_s || pathString.startsWith("[stack"_s);
                return;
            }
        }

        if (!isAnonymous)
            return;

        unsigned long privateDirtyInKB;
        if (sscanf(buffer, "Private_Dirty: %lu", &privateDirtyInKB) == 1)
            totalPrivateDirtyInKB += privateDirtyInKB;
    });
    fclose(file);
    return totalPrivateDirtyInKB * KB;
}

size_t memoryFootprint()
{
    static size_t footprint = 0;
    static MonotonicTime previousUpdateTime = { };
    Seconds elapsed = MonotonicTime::now() - previousUpdateTime;
    if (elapsed >= s_memoryFootprintUpdateInterval) {
        footprint = computeMemoryFootprint();
        previousUpdateTime = MonotonicTime::now();
    }

    return footprint;
}

} // namespace WTF
