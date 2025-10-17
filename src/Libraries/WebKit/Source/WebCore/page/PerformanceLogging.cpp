/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 21, 2023.
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
#include "PerformanceLogging.h"

#include "BackForwardCache.h"
#include "CommonVM.h"
#include "Document.h"
#include "FrameLoader.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "LocalFrameLoaderClient.h"
#include "Logging.h"
#include "Page.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PerformanceLogging);

#if !RELEASE_LOG_DISABLED
static ASCIILiteral toString(PerformanceLogging::PointOfInterest poi)
{
    switch (poi) {
    case PerformanceLogging::MainFrameLoadStarted:
        return "MainFrameLoadStarted"_s;
    case PerformanceLogging::MainFrameLoadCompleted:
        return "MainFrameLoadCompleted"_s;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return ""_s;
}
#endif

Vector<std::pair<ASCIILiteral, size_t>> PerformanceLogging::memoryUsageStatistics(ShouldIncludeExpensiveComputations includeExpensive)
{
    Vector<std::pair<ASCIILiteral, size_t>> stats;
    stats.reserveInitialCapacity(32);

    stats.append(std::pair { "page_count"_s, Page::nonUtilityPageCount() });
    stats.append(std::pair { "backforward_cache_page_count"_s, BackForwardCache::singleton().pageCount() });
    stats.append(std::pair { "document_count"_s, Document::allDocuments().size() });

    Ref vm = commonVM();
    JSC::JSLockHolder locker(vm);
    stats.append(std::pair { "javascript_gc_heap_capacity_mb"_s, vm->heap.capacity() >> 20 });
    stats.append(std::pair { "javascript_gc_heap_extra_memory_size_mb"_s, vm->heap.extraMemorySize() >> 20 });

    if (includeExpensive == ShouldIncludeExpensiveComputations::Yes) {
        stats.append(std::pair { "javascript_gc_heap_size_mb"_s, vm->heap.size() >> 20 });
        stats.append(std::pair { "javascript_gc_object_count"_s, vm->heap.objectCount() });
        stats.append(std::pair { "javascript_gc_protected_object_count"_s, vm->heap.protectedObjectCount() });
        stats.append(std::pair { "javascript_gc_protected_global_object_count"_s, vm->heap.protectedGlobalObjectCount() });
    }

    getPlatformMemoryUsageStatistics(stats);

    return stats;
}

HashCountedSet<const char*> PerformanceLogging::javaScriptObjectCounts()
{
    return WTFMove(*commonVM().heap.objectTypeCounts());
}

PerformanceLogging::PerformanceLogging(Page& page)
    : m_page(page)
{
}

void PerformanceLogging::didReachPointOfInterest(PointOfInterest poi)
{
#if RELEASE_LOG_DISABLED
    UNUSED_PARAM(poi);
    UNUSED_VARIABLE(m_page);
#else
    // Ignore synthetic main frames used internally by SVG and web inspector.
    if (auto* localMainFrame = dynamicDowncast<LocalFrame>(m_page.mainFrame())) {
        if (localMainFrame->loader().client().isEmptyFrameLoaderClient())
            return;
    }

    RELEASE_LOG(PerformanceLogging, "Memory usage info dump at %s:", toString(poi).characters());
    for (auto& [key, value] : memoryUsageStatistics(ShouldIncludeExpensiveComputations::No))
        RELEASE_LOG(PerformanceLogging, "  %s: %zu", key.characters(), value);
#endif
}

#if !PLATFORM(COCOA)
void PerformanceLogging::getPlatformMemoryUsageStatistics(Vector<std::pair<ASCIILiteral, size_t>>&) { }
std::optional<uint64_t> PerformanceLogging::physicalFootprint() { return std::nullopt; }
#endif

}
