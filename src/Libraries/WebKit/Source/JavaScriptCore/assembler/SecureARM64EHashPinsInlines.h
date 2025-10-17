/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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

#include "SecureARM64EHashPins.h"

#if CPU(ARM64E) && ENABLE(JIT)

namespace JSC {

ALWAYS_INLINE uint64_t SecureARM64EHashPins::keyForCurrentThread()
{
    uint64_t result;
    asm (
        "mrs %x[result], TPIDRRO_EL0"
        : [result] "=r" (result)
        :
        :
    );
#if !HAVE(SIMPLIFIED_FAST_TLS_BASE)
    result = result & ~0x7ull;
#endif
    return result;
}

template <typename Function>
ALWAYS_INLINE void SecureARM64EHashPins::forEachPage(Function function)
{
    Page* page = firstPage();
    do {
        RELEASE_ASSERT(isJITPC(page));
        if (function(*page) == IterationStatus::Done)
            return;
        page = page->next;
    } while (page);
}

template <typename Function>
ALWAYS_INLINE void SecureARM64EHashPins::forEachEntry(Function function)
{
    size_t baseIndex = 0;
    forEachPage([&] (Page& page) {
        IterationStatus iterationStatus = IterationStatus::Continue;
        page.forEachSetBit([&] (size_t bitIndex) {
            Entry& entry = page.fastEntryUnchecked(bitIndex);
            ASSERT(isJITPC(&entry));
            size_t index = baseIndex + bitIndex;
            iterationStatus = function(page, entry, index);
            return iterationStatus;
        });
        baseIndex += numEntriesPerPage;
        return iterationStatus;
    });
}

ALWAYS_INLINE auto SecureARM64EHashPins::findFirstEntry() -> FindResult
{
    uint64_t key = keyForCurrentThread();
    // We can call this concurrently to the bit vector being modified
    // since we either call this when we're locked, or we call it when
    // we know the entry already exists. When the entry already exists,
    // we know that the bit can't get flipped spuriously during concurrent
    // modification, so we're guaranteed to find the value. We might see
    // an entry as it's being written to, but that's also fine, since it
    // won't have our same key. It'll either be zero, or a different key.

    FindResult result;
    forEachEntry([&] (Page& page, Entry& entry, size_t index) {
        if (entry.key == key) {
            result.entry = &entry;
            result.page = &page;
            result.index = index;
            return IterationStatus::Done;
        }
        return IterationStatus::Continue;
    });

    return result;
}

ALWAYS_INLINE uint64_t SecureARM64EHashPins::pinForCurrentThread()
{
    if (LIKELY(g_jscConfig.useFastJITPermissions))
        return findFirstEntry().entry->pin;
    return 1;
}

} // namespace JSC

#endif // CPU(ARM64E) && ENABLE(JIT)
