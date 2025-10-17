/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 16, 2021.
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

#if CPU(ARM64E) && ENABLE(JIT)

#include <wtf/BitSet.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class SecureARM64EHashPins {
public:
    JS_EXPORT_PRIVATE void initializeAtStartup();
    JS_EXPORT_PRIVATE void allocatePinForCurrentThread();
    JS_EXPORT_PRIVATE void deallocatePinForCurrentThread();
    uint64_t pinForCurrentThread();

    static constexpr size_t numEntriesPerPage = 64;

    struct Entry {
        uint64_t pin;
        uint64_t key;
    };

    struct alignas(alignof(Entry)) Page {
        Page();

        static size_t allocationSize() { return sizeof(Page) + numEntriesPerPage * sizeof(Entry); }

        static constexpr size_t mask = numEntriesPerPage - 1;
        static_assert(hasOneBitSet(numEntriesPerPage));
        static_assert(!!mask);
        static_assert(!(mask & numEntriesPerPage));

        ALWAYS_INLINE Entry& entry(size_t index)
        { 
            return entries()[index & mask];
        }

        ALWAYS_INLINE Entry& fastEntryUnchecked(size_t index)
        {
            ASSERT((index & mask) == index);
            return entries()[index];
        }
        
        ALWAYS_INLINE void setIsInUse(size_t index)
        {
            isInUseMap.set(index & mask);
        }

        ALWAYS_INLINE void clearIsInUse(size_t index)
        {
            isInUseMap.set(index & mask, false);
        }

        ALWAYS_INLINE bool isInUse(size_t index)
        {
            return isInUseMap.get(index & mask);
        }

        ALWAYS_INLINE size_t findClearBit()
        {
            return isInUseMap.findBit(0, false);
        }

        template <typename Function>
        ALWAYS_INLINE void forEachSetBit(Function function)
        {
            isInUseMap.forEachSetBit(function);
        }

        Page* next { nullptr };
    private:
        Entry* entries() { return std::bit_cast<Entry*>(this + 1); }
        WTF::BitSet<numEntriesPerPage> isInUseMap;
    };

    static_assert(sizeof(Page) % alignof(Entry) == 0);

    struct alignas(alignof(Page)) Metadata {
        Atomic<uint64_t> nextPin { 1 };
        Atomic<uint32_t> assertNotReentrant { 0 };
    };

    static_assert(sizeof(Metadata) % alignof(Page) == 0);

private:
    static uint64_t keyForCurrentThread();
    bool allocatePinForCurrentThreadImpl(const AbstractLocker&);

    struct FindResult {
        Entry* entry { nullptr };
        size_t index { std::numeric_limits<size_t>::max() };
        Page* page { nullptr };
    };
    FindResult findFirstEntry();

    Metadata* metadata();
    inline Page* firstPage() { return std::bit_cast<Page*>(m_memory); }

    template <typename Function>
    void forEachPage(Function);

    template <typename Function>
    void forEachEntry(Function);

    void* m_memory { nullptr };
};

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // CPU(ARM64E) && ENABLE(JIT)
