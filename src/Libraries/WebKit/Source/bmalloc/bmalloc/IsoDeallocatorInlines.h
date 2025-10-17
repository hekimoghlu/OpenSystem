/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 9, 2024.
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

#include "BInline.h"
#include "IsoDeallocator.h"
#include "IsoPage.h"
#include "IsoSharedPage.h"
#include "Mutex.h"
#include <mutex>

#if !BUSE(LIBPAS)

namespace bmalloc {

template<typename Config>
IsoDeallocator<Config>::IsoDeallocator(Mutex& lock)
    : m_lock(&lock)
{
}

template<typename Config>
IsoDeallocator<Config>::~IsoDeallocator()
{
}

template<typename Config>
template<typename Type>
void IsoDeallocator<Config>::deallocate(api::IsoHeapBase<Type>& handle, void* ptr)
{
    static constexpr bool verbose = false;
    if (verbose)
        fprintf(stderr, "%p: deallocating %p of size %u\n", &IsoPage<Config>::pageFor(ptr)->heap(), ptr, Config::objectSize);

    // For allocation from shared pages, we deallocate immediately rather than batching deallocations with object log.
    // The batching delays the reclamation of the shared cells, which can make allocator mistakenly think that "we exhaust shared
    // cells because this is allocated a lot". Since the number of shared cells are limited, this immediate deallocation path
    // should be rarely taken. If we see frequent malloc-and-free pattern, we tier up the allocator from shared mode to fast mode.
    IsoPageBase* page = IsoPageBase::pageFor(ptr);
    if (page->isShared()) {
        LockHolder locker(*m_lock);
        static_cast<IsoSharedPage*>(page)->free<Config>(locker, handle, ptr);
        return;
    }

    if (m_objectLog.size() == m_objectLog.capacity())
        scavenge();
    
    m_objectLog.push(ptr);
}

template<typename Config>
BNO_INLINE void IsoDeallocator<Config>::scavenge()
{
    LockHolder locker(*m_lock);
    
    for (void* ptr : m_objectLog)
        IsoPage<Config>::pageFor(ptr)->free(locker, ptr);
    m_objectLog.clear();
}

} // namespace bmalloc

#endif
#endif // !BUSE(TZONE)
