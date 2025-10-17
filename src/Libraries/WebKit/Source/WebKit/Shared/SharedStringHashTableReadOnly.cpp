/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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
#include "SharedStringHashTableReadOnly.h"

#include <WebCore/SharedMemory.h>
#include <wtf/StdLibExtras.h>

namespace WebKit {

using namespace WebCore;

#if ASSERT_ENABLED
static inline bool isPowerOf2(unsigned v)
{
    // Taken from http://www.cs.utk.edu/~vose/c-stuff/bithacks.html

    return !(v & (v - 1)) && v;
}
#endif

static inline unsigned doubleHash(unsigned key)
{
    key = ~key + (key >> 23);
    key ^= (key << 12);
    key ^= (key >> 7);
    key ^= (key << 2);
    key ^= (key >> 20);
    return key;
}

SharedStringHashTableReadOnly::SharedStringHashTableReadOnly() = default;

SharedStringHashTableReadOnly::~SharedStringHashTableReadOnly() = default;

void SharedStringHashTableReadOnly::setSharedMemory(RefPtr<SharedMemory>&& sharedMemory)
{
    m_sharedMemory = WTFMove(sharedMemory);

    if (m_sharedMemory) {
        ASSERT(!(m_sharedMemory->size() % sizeof(SharedStringHash)));
        m_table = spanReinterpretCast<SharedStringHash>(m_sharedMemory->mutableSpan());
        ASSERT(isPowerOf2(m_table.size()));
        m_tableSizeMask = m_table.size() - 1;
    } else {
        m_table = { };
        m_tableSizeMask = 0;
    }
}

bool SharedStringHashTableReadOnly::contains(SharedStringHash sharedStringHash) const
{
    auto* slot = findSlot(sharedStringHash);
    return slot && *slot;
}

SharedStringHash* SharedStringHashTableReadOnly::findSlot(SharedStringHash sharedStringHash) const
{
    if (!m_sharedMemory)
        return nullptr;

    int k = 0;
    auto table = m_table;
    int sizeMask = m_tableSizeMask;
    unsigned h = static_cast<unsigned>(sharedStringHash);
    int i = h & sizeMask;

    while (1) {
        auto& entry = table[i];

        // Check if we've reached the end of the table.
        if (!entry)
            return &entry;

        if (entry == sharedStringHash)
            return &entry;

        if (!k)
            k = 1 | doubleHash(h);
        i = (i + k) & sizeMask;
    }
}

} // namespace WebKit
