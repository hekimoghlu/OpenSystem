/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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

#include <wtf/HashCountedSet.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Page;

enum class ShouldIncludeExpensiveComputations : bool { No, Yes };

class PerformanceLogging {
    WTF_MAKE_TZONE_ALLOCATED(PerformanceLogging);
    WTF_MAKE_NONCOPYABLE(PerformanceLogging);
public:
    explicit PerformanceLogging(Page&);

    enum PointOfInterest {
        MainFrameLoadStarted,
        MainFrameLoadCompleted,
    };

    void didReachPointOfInterest(PointOfInterest);

    WEBCORE_EXPORT static HashCountedSet<const char*> javaScriptObjectCounts();
    WEBCORE_EXPORT static Vector<std::pair<ASCIILiteral, size_t>> memoryUsageStatistics(ShouldIncludeExpensiveComputations);
    WEBCORE_EXPORT static std::optional<uint64_t> physicalFootprint();

private:
    static void getPlatformMemoryUsageStatistics(Vector<std::pair<ASCIILiteral, size_t>>&);

    Page& m_page;
};

}
