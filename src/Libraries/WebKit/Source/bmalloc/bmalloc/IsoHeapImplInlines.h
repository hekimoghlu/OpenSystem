/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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

#include "IsoHeapImpl.h"
#include "IsoTLSDeallocatorEntry.h"
#include "IsoSharedHeapInlines.h"
#include "IsoSharedPageInlines.h"

#if !BUSE(LIBPAS)

namespace bmalloc {

template<typename Config>
IsoHeapImpl<Config>::IsoHeapImpl()
    : IsoHeapImplBase((*PerProcess<IsoTLSEntryHolder<IsoTLSDeallocatorEntry<Config>>>::get())->lock)
    , m_inlineDirectory(*this)
    , m_allocator(*this)
{
#if BUSE(LIBPAS) && BCOMPILER(CLANG)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"
    RELEASE_BASSERT(!"Should not be using IsoHeapImpl if BUSE(LIBPAS)");
#pragma clang diagnostic pop
#endif
}

template<typename Config>
EligibilityResult<Config> IsoHeapImpl<Config>::takeFirstEligible(const LockHolder& locker)
{
    if (m_isInlineDirectoryEligibleOrDecommitted) {
        EligibilityResult<Config> result = m_inlineDirectory.takeFirstEligible(locker);
        if (result.kind == EligibilityKind::Full)
            m_isInlineDirectoryEligibleOrDecommitted = false;
        else
            return result;
    }
    
    {
        auto* cursor = m_firstEligibleOrDecommitedDirectory.get();
        if (!cursor) {
            // If nothing is eligible, it can only be because we have no directories. It wouldn't be the end
            // of the world if we broke this invariant. It would only mean that didBecomeEligibleOrDecommited() would need
            // a null check.
            RELEASE_BASSERT(!m_headDirectory.get());
            RELEASE_BASSERT(!m_tailDirectory.get());
        } else {
            auto* originalCursor = cursor;
            BUNUSED(originalCursor);
            for (; cursor; cursor = cursor->next) {
                EligibilityResult<Config> result = cursor->payload.takeFirstEligible(locker);
                // While iterating, m_firstEligibleOrDecommitedDirectory is never changed. We are holding a lock,
                // and IsoDirectory::takeFirstEligible must not populate a new eligibile / decommitted pages.
                BASSERT(m_firstEligibleOrDecommitedDirectory.get() == originalCursor);
                if (result.kind != EligibilityKind::Full) {
                    m_directoryHighWatermark = std::max(m_directoryHighWatermark, cursor->index());
                    m_firstEligibleOrDecommitedDirectory = cursor;
                    return result;
                }
            }
            m_firstEligibleOrDecommitedDirectory = nullptr;
        }
    }
    
    auto* newDirectory = new IsoDirectoryPage<Config>(*this, m_nextDirectoryPageIndex++);
    if (m_headDirectory.get()) {
        m_tailDirectory->next = newDirectory;
        m_tailDirectory = newDirectory;
    } else {
        RELEASE_BASSERT(!m_tailDirectory.get());
        m_headDirectory = newDirectory;
        m_tailDirectory = newDirectory;
    }
    m_directoryHighWatermark = newDirectory->index();
    m_firstEligibleOrDecommitedDirectory = newDirectory;
    EligibilityResult<Config> result = newDirectory->payload.takeFirstEligible(locker);
    RELEASE_BASSERT(result.kind != EligibilityKind::Full);
    return result;
}

template<typename Config>
void IsoHeapImpl<Config>::didBecomeEligibleOrDecommited(const LockHolder&, IsoDirectory<Config, numPagesInInlineDirectory>* directory)
{
    RELEASE_BASSERT(directory == &m_inlineDirectory);
    m_isInlineDirectoryEligibleOrDecommitted = true;
}

template<typename Config>
void IsoHeapImpl<Config>::didBecomeEligibleOrDecommited(const LockHolder&, IsoDirectory<Config, IsoDirectoryPage<Config>::numPages>* directory)
{
    RELEASE_BASSERT(m_firstEligibleOrDecommitedDirectory);
    auto* directoryPage = IsoDirectoryPage<Config>::pageFor(directory);
    if (directoryPage->index() < m_firstEligibleOrDecommitedDirectory->index())
        m_firstEligibleOrDecommitedDirectory = directoryPage;
}

template<typename Config>
void IsoHeapImpl<Config>::scavenge(Vector<DeferredDecommit>& decommits)
{
    LockHolder locker(this->lock);
    forEachDirectory(
        locker,
        [&] (auto& directory) {
            directory.scavenge(locker, decommits);
        });
    m_directoryHighWatermark = 0;
}

inline size_t IsoHeapImplBase::freeableMemory()
{
    return m_freeableMemory;
}

template<typename Config>
unsigned IsoHeapImpl<Config>::allocatorOffset()
{
    return m_allocator->offset();
}

template<typename Config>
unsigned IsoHeapImpl<Config>::deallocatorOffset()
{
    return (*PerProcess<IsoTLSEntryHolder<IsoTLSDeallocatorEntry<Config>>>::get())->offset();
}

template<typename Config>
unsigned IsoHeapImpl<Config>::numLiveObjects()
{
    LockHolder locker(this->lock);
    unsigned result = 0;
    forEachLiveObject(
        locker,
        [&] (void*) {
            result++;
        });
    return result;
}

template<typename Config>
unsigned IsoHeapImpl<Config>::numCommittedPages()
{
    LockHolder locker(this->lock);
    unsigned result = 0;
    forEachCommittedPage(
        locker,
        [&] (IsoPage<Config>&) {
            result++;
        });
    return result;
}

template<typename Config>
template<typename Func>
void IsoHeapImpl<Config>::forEachDirectory(const LockHolder&, const Func& func)
{
    func(m_inlineDirectory);
    for (IsoDirectoryPage<Config>* page = m_headDirectory.get(); page; page = page->next)
        func(page->payload);
}

template<typename Config>
template<typename Func>
void IsoHeapImpl<Config>::forEachCommittedPage(const LockHolder& locker, const Func& func)
{
    forEachDirectory(
        locker,
        [&] (auto& directory) {
            directory.forEachCommittedPage(locker, func);
        });
}

template<typename Config>
template<typename Func>
void IsoHeapImpl<Config>::forEachLiveObject(const LockHolder& locker, const Func& func)
{
    forEachCommittedPage(
        locker,
        [&] (IsoPage<Config>& page) {
            page.forEachLiveObject(locker, func);
        });
    for (unsigned index = 0; index < maxAllocationFromShared; ++index) {
        void* pointer = m_sharedCells[index].get();
        if (pointer && !(m_availableShared & (1U << index)))
            func(pointer);
    }
}

inline size_t IsoHeapImplBase::footprint()
{
#if ENABLE_PHYSICAL_PAGE_MAP
    RELEASE_BASSERT(m_footprint == m_physicalPageMap.footprint());
#endif
    return m_footprint;
}

inline void IsoHeapImplBase::didCommit(void* ptr, size_t bytes)
{
    BUNUSED_PARAM(ptr);
    m_footprint += bytes;
#if ENABLE_PHYSICAL_PAGE_MAP
    m_physicalPageMap.commit(ptr, bytes);
#endif
}

inline void IsoHeapImplBase::didDecommit(void* ptr, size_t bytes)
{
    BUNUSED_PARAM(ptr);
    m_footprint -= bytes;
#if ENABLE_PHYSICAL_PAGE_MAP
    m_physicalPageMap.decommit(ptr, bytes);
#endif
}

inline void IsoHeapImplBase::isNowFreeable(void* ptr, size_t bytes)
{
    BUNUSED_PARAM(ptr);
    m_freeableMemory += bytes;
}

inline void IsoHeapImplBase::isNoLongerFreeable(void* ptr, size_t bytes)
{
    BUNUSED_PARAM(ptr);
    m_freeableMemory -= bytes;
}

template<typename Config>
AllocationMode IsoHeapImpl<Config>::updateAllocationMode()
{
    auto getNewAllocationMode = [&] {
        // Exhaust shared free cells, which means we should start activating the fast allocation mode for this type.
        if (!m_availableShared) {
            m_lastSlowPathTime = std::chrono::steady_clock::now();
            return AllocationMode::Fast;
        }

        switch (m_allocationMode) {
        case AllocationMode::Shared:
            // Currently in the shared allocation mode. Until we exhaust shared free cells, continue using the shared allocation mode.
            // But if we allocate so many shared cells within very short period, we should use the fast allocation mode instead.
            // This avoids the following pathological case.
            //
            //     for (int i = 0; i < 1e6; ++i) {
            //         auto* ptr = allocate();
            //         ...
            //         free(ptr);
            //     }
            if (m_numberOfAllocationsFromSharedInOneCycle <= IsoPage<Config>::numObjects)
                return AllocationMode::Shared;
            BFALLTHROUGH;

        case AllocationMode::Fast: {
            // The allocation pattern may change. We should check the allocation rate and decide which mode is more appropriate.
            // If we don't go to the allocation slow path during ~1 seconds, we think the allocation becomes quiescent state.
            auto now = std::chrono::steady_clock::now();
            if ((now - m_lastSlowPathTime) < std::chrono::seconds(1)) {
                m_lastSlowPathTime = now;
                return AllocationMode::Fast;
            }

            m_numberOfAllocationsFromSharedInOneCycle = 0;
            m_lastSlowPathTime = now;
            return AllocationMode::Shared;
        }

        case AllocationMode::Init:
            m_lastSlowPathTime = std::chrono::steady_clock::now();
            return AllocationMode::Shared;
        }

        return AllocationMode::Shared;
    };
    AllocationMode allocationMode = getNewAllocationMode();
    m_allocationMode = allocationMode;
    return allocationMode;
}

template<typename Config>
void* IsoHeapImpl<Config>::allocateFromShared(const LockHolder&, bool abortOnFailure)
{
    static constexpr bool verbose = false;

    unsigned indexPlusOne = __builtin_ffs(m_availableShared);
    BASSERT(indexPlusOne);
    unsigned index = indexPlusOne - 1;
    void* result = m_sharedCells[index].get();
    if (result) {
        if (verbose)
            fprintf(stderr, "%p: allocated %p from shared again of size %u\n", this, result, Config::objectSize);
    } else {
        constexpr unsigned objectSizeWithHeapImplPointer = Config::objectSize + sizeof(uint8_t);
        result = IsoSharedHeap::get()->allocateNew<objectSizeWithHeapImplPointer>(abortOnFailure);
        if (!result)
            return nullptr;
        if (verbose)
            fprintf(stderr, "%p: allocated %p from shared of size %u\n", this, result, Config::objectSize);
        BASSERT(index < IsoHeapImplBase::maxAllocationFromShared);
        *indexSlotFor<Config>(result) = index;
        m_sharedCells[index] = std::bit_cast<uint8_t*>(result);
    }
    BASSERT(result);
    m_availableShared &= ~(1U << index);
    ++m_numberOfAllocationsFromSharedInOneCycle;
    return result;
}

} // namespace bmalloc

#endif
#endif // !BUSE(TZONE)
