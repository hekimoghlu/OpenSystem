/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 12, 2023.
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

#include "CPU.h"
#include <wtf/HashMap.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Threading.h>

#if ENABLE(ASSEMBLER)

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

namespace Probe {

class Page {
    WTF_MAKE_TZONE_ALLOCATED(Page);
public:
    Page(void* baseAddress);

    static void* baseAddressFor(void* p)
    {
        return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(p) & ~s_pageMask);
    }
    static void* chunkAddressFor(void* p)
    {
        return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(p) & ~s_chunkMask);
    }

    void* baseAddress() { return m_baseLogicalAddress; }

    template<typename T>
    T get(void* logicalAddress)
    {
        void* from = physicalAddressFor(logicalAddress);
        typename std::remove_const<T>::type to { };
        std::memcpy(&to, from, sizeof(to)); // Use std::memcpy to avoid strict aliasing issues.
        return to;
    }
    template<typename T>
    T get(void* logicalBaseAddress, ptrdiff_t offset)
    {
        return get<T>(static_cast<uint8_t*>(logicalBaseAddress) + offset);
    }

    template<typename T>
    void set(void* logicalAddress, T value)
    {
        if (sizeof(T) <= s_chunkSize)
            m_dirtyBits |= dirtyBitFor(logicalAddress);
        else {
            size_t numberOfChunks = roundUpToMultipleOf<sizeof(T)>(s_chunkSize) / s_chunkSize;
            uint8_t* dirtyAddress = static_cast<uint8_t*>(logicalAddress);
            for (size_t i = 0; i < numberOfChunks; ++i, dirtyAddress += s_chunkSize)
                m_dirtyBits |= dirtyBitFor(dirtyAddress);
        }
        void* to = physicalAddressFor(logicalAddress);
        std::memcpy(to, &value, sizeof(T)); // Use std::memcpy to avoid strict aliasing issues.
    }
    template<typename T>
    void set(void* logicalBaseAddress, ptrdiff_t offset, T value)
    {
        set<T>(static_cast<uint8_t*>(logicalBaseAddress) + offset, value);
    }

    bool hasWritesToFlush() const { return !!m_dirtyBits; }
    void flushWritesIfNeeded()
    {
        if (m_dirtyBits)
            flushWrites();
    }

    void* lowWatermarkFromVisitingDirtyChunks();

private:
    uint64_t dirtyBitFor(void* logicalAddress)
    {
        uintptr_t offset = reinterpret_cast<uintptr_t>(logicalAddress) & s_pageMask;
        return static_cast<uint64_t>(1) << (offset >> s_chunkSizeShift);
    }

    void* physicalAddressFor(void* logicalAddress)
    {
        return static_cast<uint8_t*>(logicalAddress) + m_physicalAddressOffset;
    }

    void flushWrites();

    void* m_baseLogicalAddress { nullptr };
    ptrdiff_t m_physicalAddressOffset;
    uint64_t m_dirtyBits { 0 };

#if ASAN_ENABLED
    // The ASan stack may contain poisoned words that may be manipulated at ASan's discretion.
    // We would never touch those words anyway, but let's ensure that the page size is set
    // such that the chunk size is guaranteed to be exactly sizeof(uintptr_t) so that we won't
    // inadvertently overwrite one of ASan's words on the stack when we copy back the dirty
    // chunks.
    // FIXME: we should consider using the same page size for both ASan and non-ASan builds.
    // https://bugs.webkit.org/show_bug.cgi?id=176961
    static constexpr size_t s_pageSize = 64 * sizeof(uintptr_t); // because there are 64 bits in m_dirtyBits.
#else // not ASAN_ENABLED
    static constexpr size_t s_pageSize = 1024;
#endif // ASAN_ENABLED
    static constexpr uintptr_t s_pageMask = s_pageSize - 1;
    static constexpr size_t s_chunksPerPage = sizeof(uint64_t) * 8; // number of bits in m_dirtyBits.
    static constexpr size_t s_chunkSize = s_pageSize / s_chunksPerPage;
    static constexpr uintptr_t s_chunkMask = s_chunkSize - 1;
#if ASAN_ENABLED
    static_assert(s_chunkSize == sizeof(uintptr_t), "bad chunkSizeShift");
    static constexpr size_t s_chunkSizeShift = is64Bit() ? 3 : 2;
#else // no ASAN_ENABLED
    static constexpr size_t s_chunkSizeShift = 4;
#endif // ASAN_ENABLED
    static_assert(s_pageSize > s_chunkSize, "bad pageSize or chunkSize");
    static_assert(s_chunkSize == (1 << s_chunkSizeShift), "bad chunkSizeShift");

    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    typedef typename std::aligned_storage<s_pageSize, std::alignment_of<uintptr_t>::value>::type Buffer;
    ALLOW_DEPRECATED_DECLARATIONS_END
    Buffer m_buffer;
};

class Stack {
    WTF_MAKE_TZONE_ALLOCATED(Stack);
public:
    Stack()
        : m_stackBounds(Thread::current().stack())
    { }
    Stack(Stack&& other);

    void* lowWatermarkFromVisitingDirtyPages();
    void* lowWatermark(void* stackPointer)
    {
        ASSERT(Page::chunkAddressFor(stackPointer) == lowWatermarkFromVisitingDirtyPages());
        return Page::chunkAddressFor(stackPointer);
    }

    template<typename T>
    T get(void* address)
    {
        Page* page = pageFor(address);
        return page->get<T>(address);
    }
    template<typename T>
    T get(void* logicalBaseAddress, ptrdiff_t offset)
    {
        return get<T>(static_cast<uint8_t*>(logicalBaseAddress) + offset);
    }

    template<typename T>
    void set(void* address, T value)
    {
        Page* page = pageFor(address);
        page->set<T>(address, value);
    }

    template<typename T>
    void set(void* logicalBaseAddress, ptrdiff_t offset, T value)
    {
        set<T>(static_cast<uint8_t*>(logicalBaseAddress) + offset, value);
    }

    JS_EXPORT_PRIVATE Page* ensurePageFor(void* address);

    void* savedStackPointer() const { return m_savedStackPointer; }
    void setSavedStackPointer(void* sp) { m_savedStackPointer = sp; }

    bool hasWritesToFlush();
    void flushWrites();

#if ASSERT_ENABLED
    bool isValid() { return m_isValid; }
#endif

private:
    Page* pageFor(void* address)
    {
        if (LIKELY(Page::baseAddressFor(address) == m_lastAccessedPageBaseAddress))
            return m_lastAccessedPage;
        return ensurePageFor(address);
    }

    void* m_savedStackPointer { nullptr };

    // A cache of the last accessed page details for quick access.
    void* m_lastAccessedPageBaseAddress { nullptr };
    Page* m_lastAccessedPage { nullptr };

    StackBounds m_stackBounds;
    UncheckedKeyHashMap<void*, std::unique_ptr<Page>> m_pages;

#if ASSERT_ENABLED
    bool m_isValid { true };
#endif
};

} // namespace Probe
} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(ASSEMBLER)
