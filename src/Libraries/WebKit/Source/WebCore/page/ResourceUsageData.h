/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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

#if ENABLE(RESOURCE_USAGE)

#include <array>
#include <wtf/MonotonicTime.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

// v(name, id, subcategory)
#define WEBCORE_EACH_MEMORY_CATEGORIES(v) \
    v(bmalloc, 0, false) \
    v(LibcMalloc, 1, false) \
    v(JSJIT, 2, false) \
    v(Gigacage, 3, false) \
    v(Images, 4, false) \
    v(GCHeap, 5, true) \
    v(GCOwned, 6, true) \
    v(Other, 7, false) \
    v(Layers, 8, false) \
    v(IsoHeap, 9, false) \

namespace MemoryCategory {
#define WEBCORE_DEFINE_MEMORY_CATEGORY(name, id, subcategory) static constexpr unsigned name = id;
WEBCORE_EACH_MEMORY_CATEGORIES(WEBCORE_DEFINE_MEMORY_CATEGORY)
#undef WEBCORE_DEFINE_MEMORY_CATEGORY

#define WEBCORE_DEFINE_MEMORY_CATEGORY(name, id, subcategory) + 1
static constexpr unsigned NumberOfCategories = 0 WEBCORE_EACH_MEMORY_CATEGORIES(WEBCORE_DEFINE_MEMORY_CATEGORY);
#undef WEBCORE_DEFINE_MEMORY_CATEGORY
}

struct MemoryCategoryInfo {
    constexpr MemoryCategoryInfo() = default; // Needed for std::array.
    constexpr MemoryCategoryInfo(unsigned category, bool subcategory = false)
        : isSubcategory(subcategory)
        , type(category)
    {
    }

    size_t totalSize() const { return dirtySize + externalSize; }

    size_t dirtySize { 0 };
    size_t reclaimableSize { 0 };
    size_t externalSize { 0 };
    bool isSubcategory { false };
    unsigned type { MemoryCategory::NumberOfCategories };
};

struct ThreadCPUInfo {
    enum class Type : uint8_t {
        Unknown,
        Main,
        WebKit,
    };

    String name;
    String identifier;
    float cpu { 0 };
    Type type { ThreadCPUInfo::Type::Unknown };
};

struct ResourceUsageData {
    ResourceUsageData() = default;

    float cpu { 0 };
    float cpuExcludingDebuggerThreads { 0 };
    Vector<ThreadCPUInfo> cpuThreads;

    size_t totalDirtySize { 0 };
    size_t totalExternalSize { 0 };
    std::array<MemoryCategoryInfo, MemoryCategory::NumberOfCategories> categories { {
#define WEBCORE_DEFINE_MEMORY_CATEGORY(name, id, subcategory) MemoryCategoryInfo { MemoryCategory::name, subcategory },
WEBCORE_EACH_MEMORY_CATEGORIES(WEBCORE_DEFINE_MEMORY_CATEGORY)
#undef WEBCORE_DEFINE_MEMORY_CATEGORY
    } };
    MonotonicTime timestamp { MonotonicTime::now() };
    MonotonicTime timeOfNextEdenCollection { MonotonicTime::nan() };
    MonotonicTime timeOfNextFullCollection { MonotonicTime::nan() };
};

} // namespace WebCore

#endif // ResourceUsageData_h
