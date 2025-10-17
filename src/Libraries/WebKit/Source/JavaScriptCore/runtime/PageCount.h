/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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

#include <algorithm>
#include <limits.h>

namespace WTF {
class PrintStream;
}

namespace JSC {

#if USE(LARGE_TYPED_ARRAYS)
static_assert(sizeof(size_t) == sizeof(uint64_t));
#define MAX_ARRAY_BUFFER_SIZE (1ull << 32)
#else
static_assert(sizeof(size_t) == sizeof(uint32_t));
// Because we are using a size_t to store the size in bytes of array buffers, we cannot support 4GB on 32-bit platforms.
// So we are sticking with 2GB. It should in theory be possible to support up to (4GB - 1B) if anyone cares.
#define MAX_ARRAY_BUFFER_SIZE 0x7fffffffu
#endif

class PageCount {
public:
    PageCount()
        : m_pageCount(UINT_MAX)
    {
        static_assert(maxPageCount < UINT_MAX, "We rely on this here.");
    }

    PageCount(uint32_t pageCount)
        : m_pageCount(pageCount)
    { }

    void dump(WTF::PrintStream&) const;

    uint64_t bytes() const { return static_cast<uint64_t>(m_pageCount) * static_cast<uint64_t>(pageSize); }
    uint32_t pageCount() const { return m_pageCount; }

    static bool isValid(uint32_t pageCount)
    {
        return pageCount <= maxPageCount;
    }
    
    bool isValid() const
    {
        return isValid(m_pageCount);
    }

    static PageCount fromBytes(uint64_t bytes)
    {
        RELEASE_ASSERT(bytes % pageSize == 0);
        uint32_t numPages = bytes / pageSize;
        RELEASE_ASSERT(PageCount::isValid(numPages));
        return PageCount(numPages);
    }

    static PageCount fromBytesWithRoundUp(uint64_t bytes)
    {
        return fromBytes(roundUpToMultipleOf<pageSize>(bytes));
    }

    static PageCount max()
    {
        return PageCount(maxPageCount);
    }

    explicit operator bool() const
    {
        return m_pageCount != UINT_MAX;
    }

    bool operator<(const PageCount& other) const { return m_pageCount < other.m_pageCount; }
    bool operator>(const PageCount& other) const { return m_pageCount > other.m_pageCount; }
    bool operator>=(const PageCount& other) const { return m_pageCount >= other.m_pageCount; }
    friend bool operator==(const PageCount&, const PageCount&) = default;
    PageCount operator+(const PageCount& other) const
    {
        if (sumOverflows<uint32_t>(m_pageCount, other.m_pageCount))
            return PageCount();
        uint32_t newCount = m_pageCount + other.m_pageCount;
        if (!PageCount::isValid(newCount))
            return PageCount();
        return PageCount(newCount);
    }

    static constexpr uint32_t pageSize = 64 * KB;
private:
    // The spec requires we are able to instantiate a memory with a *maximum* size of 64K pages.
    // This does not mean the memory can necessarily grow that big, and where the
    // MAX_ARRAY_BUFFER_SIZE is smaller (e.g.: on 32-bit platforms), trying to grow the memory
    // that large will fail, which is acceptable according to the spec. Nevertheless, we should
    // be able to parse such a memory and instantiate it with a smaller initial size.
    static constexpr uint32_t maxPageCount = std::max<uint32_t>(64*1024, MAX_ARRAY_BUFFER_SIZE / static_cast<uint64_t>(pageSize));

    uint32_t m_pageCount;
};

} // namespace JSC
