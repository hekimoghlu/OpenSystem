/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 3, 2024.
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
#include "IsoTLSLayout.h"

#include "IsoTLSEntry.h"

namespace bmalloc {

IsoTLSLayout::IsoTLSLayout(const std::lock_guard<Mutex>&)
{
}

void IsoTLSLayout::add(IsoTLSEntry* entry)
{
    static Mutex addingMutex;
    RELEASE_BASSERT(!entry->m_next);
    std::lock_guard<Mutex> locking(addingMutex);
    if (m_head) {
        RELEASE_BASSERT(m_tail);
        entry->m_offset = roundUpToMultipleOf(entry->alignment(), m_tail->extent());
        m_tail->m_next = entry;
        m_tail = entry;
    } else {
        RELEASE_BASSERT(!m_tail);
        entry->m_offset = 0;
        m_head = entry;
        m_tail = entry;
    }
}

} // namespace bmalloc

