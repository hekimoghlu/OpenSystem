/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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

#if !BUSE(TZONE)

#include "IsoTLSEntry.h"

#if !BUSE(LIBPAS)

namespace bmalloc {

template<typename Func>
void IsoTLSEntry::walkUpToInclusive(IsoTLSEntry* last, const Func& func)
{
    IsoTLSEntry* current = this;
    for (;;) {
        func(current);
        if (current == last)
            return;
        current = current->m_next;
    }
}

template<typename EntryType>
DefaultIsoTLSEntry<EntryType>::DefaultIsoTLSEntry()
    : IsoTLSEntry(sizeof(EntryType))
{
    static_assert(sizeof(EntryType) <= UINT32_MAX);
    static_assert(sizeof(void*) == alignof(EntryType), "Because IsoTLSEntry includes vtable, it should be the same to the pointer");
}

template<typename EntryType>
void DefaultIsoTLSEntry<EntryType>::move(void* passedSrc, void* dst)
{
    EntryType* src = static_cast<EntryType*>(passedSrc);
    new (dst) EntryType(std::move(*src));
    src->~EntryType();
}

template<typename EntryType>
void DefaultIsoTLSEntry<EntryType>::destruct(void* passedEntry)
{
    EntryType* entry = static_cast<EntryType*>(passedEntry);
    entry->~EntryType();
}

} // namespace bmalloc

#endif
#endif // !BUSE(TZONE)
