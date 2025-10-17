/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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

#include "IsoHeapImpl.h"
#include "IsoTLSDeallocatorEntry.h"

namespace bmalloc {

template<typename Config>
IsoHeapImpl<Config>::IsoHeapImpl()
    : lock(PerProcess<IsoTLSDeallocatorEntry<Config>>::get()->lock)
    , m_inlineDirectory(*this)
    , m_allocator(*this)
{
    addToAllIsoHeaps();
}

template<typename Config>
EligibilityResult<Config> IsoHeapImpl<Config>::takeFirstEligible()
{
    if (m_isInlineDirectoryEligible) {
        EligibilityResult<Config> result = m_inlineDirectory.takeFirstEligible();
        if (result.kind == EligibilityKind::Full)
            m_isInlineDirectoryEligible = false;
        else
            return result;
    }
    
    if (!m_firstEligibleDirectory) {
        // If nothing is eligible, it can only be because we have no directories. It wouldn't be the end
        // of the world if we broke this invariant. It would only mean that didBecomeEligible() would need
        // a null check.
        RELEASE_BASSERT(!m_headDirectory);
        RELEASE_BASSERT(!m_tailDirectory);
    }
    
    for (; m_firstEligibleDirectory; m_firstEligibleDirectory = m_firstEligibleDirectory->next) {
        EligibilityResult<Config> result = m_firstEligibleDirectory->payload.takeFirstEligible();
        if (result.kind != EligibilityKind::Full) {
            m_directoryHighWatermark = std::max(m_directoryHighWatermark, m_firstEligibleDirectory->index());
            return result;
        }
    }
    
    auto* newDirectory = new IsoDirectoryPage<Config>(*this, m_nextDirectoryPageIndex++);
    if (m_headDirectory) {
        m_tailDirectory->next = newDirectory;
        m_tailDirectory = newDirectory;
    } else {
        RELEASE_BASSERT(!m_tailDirectory);
        m_headDirectory = newDirectory;
        m_tailDirectory = newDirectory;
    }
    m_directoryHighWatermark = newDirectory->index();
    m_firstEligibleDirectory = newDirectory;
    EligibilityResult<Config> result = newDirectory->payload.takeFirstEligible();
    RELEASE_BASSERT(result.kind != EligibilityKind::Full);
    return result;
}

template<typename Config>
void IsoHeapImpl<Config>::didBecomeEligible(IsoDirectory<Config, numPagesInInlineDirectory>* directory)
{
    RELEASE_BASSERT(directory == &m_inlineDirectory);
    m_isInlineDirectoryEligible = true;
}

template<typename Config>
void IsoHeapImpl<Config>::didBecomeEligible(IsoDirectory<Config, IsoDirectoryPage<Config>::numPages>* directory)
{
    RELEASE_BASSERT(m_firstEligibleDirectory);
    auto* directoryPage = IsoDirectoryPage<Config>::pageFor(directory);
    if (directoryPage->index() < m_firstEligibleDirectory->index())
        m_firstEligibleDirectory = directoryPage;
}

template<typename Config>
void IsoHeapImpl<Config>::scavenge(Vector<DeferredDecommit>& decommits)
{
    std::lock_guard<Mutex> locker(this->lock);
    forEachDirectory(
        [&] (auto& directory) {
            directory.scavenge(decommits);
        });
    m_directoryHighWatermark = 0;
}

template<typename Config>
void IsoHeapImpl<Config>::scavengeToHighWatermark(Vector<DeferredDecommit>& decommits)
{
    std::lock_guard<Mutex> locker(this->lock);
    if (!m_directoryHighWatermark)
        m_inlineDirectory.scavengeToHighWatermark(decommits);
    for (IsoDirectoryPage<Config>* page = m_headDirectory; page; page = page->next) {
        if (page->index() >= m_directoryHighWatermark)
            page->payload.scavengeToHighWatermark(decommits);
    }
    m_directoryHighWatermark = 0;
}

template<typename Config>
size_t IsoHeapImpl<Config>::freeableMemory()
{
    return m_freeableMemory;
}

template<typename Config>
unsigned IsoHeapImpl<Config>::allocatorOffset()
{
    return m_allocator.offset();
}

template<typename Config>
unsigned IsoHeapImpl<Config>::deallocatorOffset()
{
    return PerProcess<IsoTLSDeallocatorEntry<Config>>::get()->offset();
}

template<typename Config>
unsigned IsoHeapImpl<Config>::numLiveObjects()
{
    unsigned result = 0;
    forEachLiveObject(
        [&] (void*) {
            result++;
        });
    return result;
}

template<typename Config>
unsigned IsoHeapImpl<Config>::numCommittedPages()
{
    unsigned result = 0;
    forEachCommittedPage(
        [&] (IsoPage<Config>&) {
            result++;
        });
    return result;
}

template<typename Config>
template<typename Func>
void IsoHeapImpl<Config>::forEachDirectory(const Func& func)
{
    func(m_inlineDirectory);
    for (IsoDirectoryPage<Config>* page = m_headDirectory; page; page = page->next)
        func(page->payload);
}

template<typename Config>
template<typename Func>
void IsoHeapImpl<Config>::forEachCommittedPage(const Func& func)
{
    forEachDirectory(
        [&] (auto& directory) {
            directory.forEachCommittedPage(func);
        });
}

template<typename Config>
template<typename Func>
void IsoHeapImpl<Config>::forEachLiveObject(const Func& func)
{
    forEachCommittedPage(
        [&] (IsoPage<Config>& page) {
            page.forEachLiveObject(func);
        });
}

template<typename Config>
size_t IsoHeapImpl<Config>::footprint()
{
#if ENABLE_PHYSICAL_PAGE_MAP
    RELEASE_BASSERT(m_footprint == m_physicalPageMap.footprint());
#endif
    return m_footprint;
}

template<typename Config>
void IsoHeapImpl<Config>::didCommit(void* ptr, size_t bytes)
{
    BUNUSED_PARAM(ptr);
    m_footprint += bytes;
#if ENABLE_PHYSICAL_PAGE_MAP
    m_physicalPageMap.commit(ptr, bytes);
#endif
}

template<typename Config>
void IsoHeapImpl<Config>::didDecommit(void* ptr, size_t bytes)
{
    BUNUSED_PARAM(ptr);
    m_footprint -= bytes;
#if ENABLE_PHYSICAL_PAGE_MAP
    m_physicalPageMap.decommit(ptr, bytes);
#endif
}

template<typename Config>
void IsoHeapImpl<Config>::isNowFreeable(void* ptr, size_t bytes)
{
    BUNUSED_PARAM(ptr);
    m_freeableMemory += bytes;
}

template<typename Config>
void IsoHeapImpl<Config>::isNoLongerFreeable(void* ptr, size_t bytes)
{
    BUNUSED_PARAM(ptr);
    m_freeableMemory -= bytes;
}

} // namespace bmalloc

