/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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

#include "BPlatform.h"
#include "Sizes.h"

namespace bmalloc {

BEXPORT size_t availableMemory();

#if BPLATFORM(IOS_FAMILY) || BOS(LINUX) || BOS(FREEBSD)
struct MemoryStatus {
    MemoryStatus(size_t memoryFootprint, double percentAvailableMemoryInUse)
        : memoryFootprint(memoryFootprint)
        , percentAvailableMemoryInUse(percentAvailableMemoryInUse)
    {
    }

    size_t memoryFootprint;
    double percentAvailableMemoryInUse;
};

BEXPORT MemoryStatus memoryStatus();

inline size_t memoryFootprint()
{
    auto memoryUse = memoryStatus();
    return memoryUse.memoryFootprint;
}

inline double percentAvailableMemoryInUse()
{
    auto memoryUse = memoryStatus();
    return memoryUse.percentAvailableMemoryInUse;
}
#endif

inline bool isUnderMemoryPressure()
{
#if BPLATFORM(IOS_FAMILY) || BOS(LINUX) || BOS(FREEBSD)
    return percentAvailableMemoryInUse() > memoryPressureThreshold;
#else
    return false;
#endif
}
    
} // namespace bmalloc
