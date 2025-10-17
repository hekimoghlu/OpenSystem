/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 20, 2025.
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
#include "ProbeStack.h"

#include <memory>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(ASSEMBLER)

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {
namespace Probe {

static void* const maxLowWatermark = reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max());

#if ASAN_ENABLED
// FIXME: we should consider using the copy function for both ASan and non-ASan builds.
// https://bugs.webkit.org/show_bug.cgi?id=176961
SUPPRESS_ASAN
static void copyStackPage(void* dst, void* src, size_t size)
{
    ASSERT(roundUpToMultipleOf<sizeof(uintptr_t)>(dst) == dst);
    ASSERT(roundUpToMultipleOf<sizeof(uintptr_t)>(src) == src);
    
    uintptr_t* dstPointer = reinterpret_cast<uintptr_t*>(dst);
    uintptr_t* srcPointer = reinterpret_cast<uintptr_t*>(src);
    for (; size; size -= sizeof(uintptr_t))
        *dstPointer++ = *srcPointer++;
}
#else
#define copyStackPage(dst, src, size) std::memcpy(dst, src, size)
#endif

WTF_MAKE_TZONE_ALLOCATED_IMPL(Page);

Page::Page(void* baseAddress)
    : m_baseLogicalAddress(baseAddress)
    , m_physicalAddressOffset(reinterpret_cast<uint8_t*>(&m_buffer) - reinterpret_cast<uint8_t*>(baseAddress))
{
    copyStackPage(&m_buffer, baseAddress, s_pageSize);
}

void Page::flushWrites()
{
    uint64_t dirtyBits = m_dirtyBits;
    size_t offset = 0;
    while (dirtyBits) {
        // Find start.
        if (dirtyBits & 1) {
            size_t startOffset = offset;
            // Find end.
            do {
                dirtyBits = dirtyBits >> 1;
                offset += s_chunkSize;
            } while (dirtyBits & 1);

            size_t size = offset - startOffset;
            uint8_t* src = reinterpret_cast<uint8_t*>(&m_buffer) + startOffset;
            uint8_t* dst = reinterpret_cast<uint8_t*>(m_baseLogicalAddress) + startOffset;
            copyStackPage(dst, src, size);
        }
        dirtyBits = dirtyBits >> 1;
        offset += s_chunkSize;
    }
    m_dirtyBits = 0;
}

void* Page::lowWatermarkFromVisitingDirtyChunks()
{
    uint64_t dirtyBits = m_dirtyBits;
    size_t offset = 0;
    while (dirtyBits) {
        if (dirtyBits & 1)
            return reinterpret_cast<uint8_t*>(m_baseLogicalAddress) + offset;
        dirtyBits = dirtyBits >> 1;
        offset += s_chunkSize;
    }
    return maxLowWatermark;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(Stack);

Stack::Stack(Stack&& other)
    : m_stackBounds(WTFMove(other.m_stackBounds))
    , m_pages(WTFMove(other.m_pages))
{
    m_savedStackPointer = other.m_savedStackPointer;
#if ASSERT_ENABLED
    other.m_isValid = false;
#endif
}

bool Stack::hasWritesToFlush()
{
    return std::any_of(m_pages.begin(), m_pages.end(), [] (auto& it) { return it.value->hasWritesToFlush(); });
}

void Stack::flushWrites()
{
    for (auto it = m_pages.begin(); it != m_pages.end(); ++it)
        it->value->flushWritesIfNeeded();
}

Page* Stack::ensurePageFor(void* address)
{
    // Since the machine stack is always allocated in units of whole pages, asserting
    // that the address is contained in the stack is sufficient to infer that its page
    // is also contained in the stack.
    RELEASE_ASSERT(m_stackBounds.contains(address));

    // We may have gotten here because of a cache miss. So, look up the page first
    // before allocating a new one,
    void* baseAddress = Page::baseAddressFor(address);
    auto it = m_pages.find(baseAddress);
    if (LIKELY(it != m_pages.end()))
        m_lastAccessedPage = it->value.get();
    else {
        std::unique_ptr<Page> page = makeUnique<Page>(baseAddress);
        auto result = m_pages.add(baseAddress, WTFMove(page));
        m_lastAccessedPage = result.iterator->value.get();
    }
    m_lastAccessedPageBaseAddress = baseAddress;
    return m_lastAccessedPage;
}

void* Stack::lowWatermarkFromVisitingDirtyPages()
{
    void* low = maxLowWatermark;
    for (auto it = m_pages.begin(); it != m_pages.end(); ++it) {
        Page& page = *it->value;
        if (!page.hasWritesToFlush() || low < page.baseAddress())
            continue;
        low = std::min(low, page.lowWatermarkFromVisitingDirtyChunks());
    }
    return low;
}

} // namespace Probe
} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(ASSEMBLER)
