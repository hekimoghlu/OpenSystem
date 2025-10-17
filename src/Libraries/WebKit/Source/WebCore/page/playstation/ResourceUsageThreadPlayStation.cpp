/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
#include <memory-extra/showmap.h>

namespace WebCore {

static float cpuUsage()
{
    // FIXME: Need a way to calculate cpu usage
    return 0;
}

void ResourceUsageThread::platformSaveStateBeforeStarting()
{
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
    auto& categories = data.categories;

    auto currentGCHeapCapacity = vm->heap.blockBytesAllocated();
    auto currentGCOwnedExtra = vm->heap.extraMemorySize();
    auto currentGCOwnedExternal = vm->heap.externalMemorySize();
    RELEASE_ASSERT(currentGCOwnedExternal <= currentGCOwnedExtra);

    categories[MemoryCategory::GCHeap].dirtySize = currentGCHeapCapacity;
    categories[MemoryCategory::GCOwned].dirtySize = currentGCOwnedExtra - currentGCOwnedExternal;
    categories[MemoryCategory::GCOwned].externalSize = currentGCOwnedExternal;

    auto currentGCDirtySize = currentGCHeapCapacity + currentGCOwnedExtra - currentGCOwnedExternal;

    // TODO: collect dirty size of "MemoryCategory::Images" and "MemoryCategory::Layers"

    memory_extra::showmap::Result<4> result;
    auto entry = result.reserve("SceNKFastMalloc");
    result.collect();
    data.totalDirtySize = result.rss;

    auto rss = entry->rss;
    RELEASE_ASSERT(data.totalDirtySize > rss);
    categories[MemoryCategory::Other].dirtySize = data.totalDirtySize - rss;
    categories[MemoryCategory::bmalloc].dirtySize = rss - std::min(rss, currentGCDirtySize);

    data.totalExternalSize = currentGCOwnedExternal;

    data.timeOfNextEdenCollection = data.timestamp + vm->heap.edenActivityCallback()->timeUntilFire().value_or(Seconds(std::numeric_limits<double>::infinity()));
    data.timeOfNextFullCollection = data.timestamp + vm->heap.fullActivityCallback()->timeUntilFire().value_or(Seconds(std::numeric_limits<double>::infinity()));
}

} // namespace WebCore

#endif // ENABLE(RESOURCE_USAGE)
