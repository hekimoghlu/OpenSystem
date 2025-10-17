/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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
#include "BPlatform.h"
#include "IsoTLSLayout.h"

#if !BUSE(TZONE)

#include "IsoTLSEntry.h"

#if !BUSE(LIBPAS)

namespace bmalloc {

DEFINE_STATIC_PER_PROCESS_STORAGE(IsoTLSLayout);

IsoTLSLayout::IsoTLSLayout(const LockHolder&)
{
}

void IsoTLSLayout::add(IsoTLSEntry* entry)
{
    static Mutex addingMutex;
    RELEASE_BASSERT(!entry->m_next);
    // IsoTLSLayout::head() does not take a lock. So we should emit memory fence to make sure that newly added entry is initialized when it is chained to this linked-list.
    // Emitting memory fence here is OK since this function is not frequently called.
    LockHolder locking(addingMutex);
    if (m_head) {
        RELEASE_BASSERT(m_tail);
        size_t offset = roundUpToMultipleOf(entry->alignment(), m_tail->extent());
        RELEASE_BASSERT(offset < UINT_MAX);
        entry->m_offset = offset;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        m_tail->m_next = entry;
        m_tail = entry;
    } else {
        RELEASE_BASSERT(!m_tail);
        entry->m_offset = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        m_head = entry;
        m_tail = entry;
    }
}

} // namespace bmalloc

#endif
#endif // !BUSE(TZONE)
