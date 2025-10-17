/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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
#include "ResourceUsageThread.h"

#if ENABLE(RESOURCE_USAGE)

#include <JavaScriptCore/GCActivityCallback.h>
#include <JavaScriptCore/VM.h>
#include <psapi.h>

namespace WebCore {

void ResourceUsageThread::platformSaveStateBeforeStarting()
{
}

static uint64_t fileTimeToUint64(FILETIME ft)
{
    ULARGE_INTEGER u;
    u.LowPart = ft.dwLowDateTime;
    u.HighPart = ft.dwHighDateTime;

    return u.QuadPart;
}

static bool getCurrentCpuTime(uint64_t& nowTime, uint64_t& userTime, uint64_t& kernelTime)
{
    FILETIME creationFileTime, exitFileTime, kernelFileTime, userFileTime;
    if (!GetProcessTimes(GetCurrentProcess(), &creationFileTime, &exitFileTime, &kernelFileTime, &userFileTime))
        return false;

    FILETIME nowFileTime;
    GetSystemTimeAsFileTime(&nowFileTime);

    nowTime = fileTimeToUint64(nowFileTime);
    userTime = fileTimeToUint64(userFileTime);
    kernelTime  = fileTimeToUint64(kernelFileTime);

    return true;
}

static float cpuUsage()
{
    static int numberOfProcessors = 0;
    static uint64_t lastTime = 0;
    static uint64_t lastKernelTime = 0;
    static uint64_t lastUserTime = 0;

    if (!lastTime) {
        SYSTEM_INFO systemInfo;
        GetSystemInfo(&systemInfo);
        numberOfProcessors = systemInfo.dwNumberOfProcessors;

        getCurrentCpuTime(lastTime, lastKernelTime, lastUserTime);
        return 0;
    }

    uint64_t nowTime, kernelTime, userTime;
    if (!getCurrentCpuTime(nowTime, kernelTime, userTime))
        return 0;

    uint64_t elapsed = nowTime - lastTime;
    uint64_t totalCPUTime = (kernelTime - lastKernelTime);
    totalCPUTime += (userTime - lastUserTime);

    lastTime = nowTime;
    lastKernelTime = kernelTime;
    lastUserTime = userTime;

    float usage =  (100.0 * totalCPUTime) / elapsed / numberOfProcessors;
    return clampTo<float>(usage, 0, 100);
}

static size_t memoryUsage()
{
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc)))
        return pmc.PrivateUsage;

    return 0;
}

void ResourceUsageThread::platformCollectCPUData(JSC::VM*, ResourceUsageData& data)
{
    data.cpu = cpuUsage();

    // FIXME: Exclude the ResourceUsage thread.
    // FIXME: Exclude the SamplingProfiler thread.
    // FIXME: Classify usage per thread.
    data.cpuExcludingDebuggerThreads = data.cpu;
}

void ResourceUsageThread::platformCollectMemoryData(JSC::VM* vm, ResourceUsageData& data)
{
    auto usage = data.totalDirtySize = memoryUsage();

    size_t currentGCHeapCapacity = vm->heap.blockBytesAllocated();
    size_t currentGCOwnedExtra = vm->heap.extraMemorySize();
    size_t currentGCOwnedExternal = vm->heap.externalMemorySize();
    RELEASE_ASSERT(currentGCOwnedExternal <= currentGCOwnedExtra);

    data.categories[MemoryCategory::GCHeap].dirtySize = currentGCHeapCapacity;
    data.categories[MemoryCategory::GCOwned].dirtySize = currentGCOwnedExtra - currentGCOwnedExternal;
    data.categories[MemoryCategory::GCOwned].externalSize = currentGCOwnedExternal;

    usage -= currentGCHeapCapacity;
    // Following ResourceUsageThreadCocoa implementation
    auto currentGCOwnedGenerallyInMalloc = currentGCOwnedExtra - currentGCOwnedExternal;
    if (currentGCOwnedGenerallyInMalloc < usage)
        usage -= currentGCOwnedGenerallyInMalloc;

    data.categories[MemoryCategory::LibcMalloc].dirtySize = usage;

    data.totalExternalSize = currentGCOwnedExternal;

    data.timeOfNextEdenCollection = data.timestamp + vm->heap.edenActivityCallback()->timeUntilFire().value_or(Seconds(std::numeric_limits<double>::infinity()));
    data.timeOfNextFullCollection = data.timestamp + vm->heap.fullActivityCallback()->timeUntilFire().value_or(Seconds(std::numeric_limits<double>::infinity()));
}

}

#endif
