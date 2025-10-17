/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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

#include "IsoPage.h"
#include "VMAllocate.h"
#include <bit>

#if !BUSE(LIBPAS)

namespace bmalloc {

// IsoSharedPage never becomes empty state again after we allocate some cells from IsoSharedPage. This makes IsoSharedPage management super simple.
// This is because empty IsoSharedPage is still split into various different objects that should keep some part of virtual memory region dedicated.
// We cannot set up bump allocation for such a page. Not freeing IsoSharedPages are OK since IsoSharedPage is only used for the lower tier of IsoHeap.
template<typename Config, typename Type>
void IsoSharedPage::free(const LockHolder&, api::IsoHeapBase<Type>& handle, void* ptr)
{
    auto& heapImpl = handle.impl();
    uint8_t index = *indexSlotFor<Config>(ptr) & IsoHeapImplBase::maxAllocationFromSharedMask;
    // IsoDeallocator::deallocate is called from delete operator. This is dispatched by vtable if virtual destructor exists.
    // If vptr is replaced to the other vptr, we may accidentally chain this pointer to the incorrect HeapImplBase, which totally breaks the IsoHeap's goal.
    // To harden that, we validate that this pointer is actually allocated for a specific HeapImplBase here by checking whether this pointer is listed in HeapImplBase's shared cells.
    RELEASE_BASSERT(heapImpl.m_sharedCells[index].get() == ptr);
    heapImpl.m_availableShared |= (1U << index);
}

inline VariadicBumpAllocator IsoSharedPage::startAllocating(const LockHolder&)
{
    static constexpr bool verbose = false;

    if (verbose) {
        fprintf(stderr, "%p: starting shared allocation.\n", this);
        fprintf(stderr, "%p: preparing to shared bump.\n", this);
    }

    char* payloadEnd = reinterpret_cast<char*>(this) + IsoSharedPage::pageSize;
    unsigned remaining = static_cast<unsigned>(roundDownToMultipleOf<alignmentForIsoSharedAllocation>(static_cast<uintptr_t>(IsoSharedPage::pageSize - sizeof(IsoSharedPage))));

    return VariadicBumpAllocator(payloadEnd, remaining);
}

inline void IsoSharedPage::stopAllocating(const LockHolder&)
{
    static constexpr bool verbose = false;

    if (verbose)
        fprintf(stderr, "%p: stopping shared allocation.\n", this);
}

} // namespace bmalloc

#endif
#endif // !BUSE(TZONE)
