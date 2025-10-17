/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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

#include "BMalloced.h"
#include "IsoAllocator.h"
#include "IsoDirectoryPage.h"
#include "IsoTLSAllocatorEntry.h"
#include "Packed.h"
#include "PhysicalPageMap.h"

#if !BUSE(LIBPAS)

namespace bmalloc {

class AllIsoHeaps;

class BEXPORT IsoHeapImplBase {
    MAKE_BMALLOCED;
    IsoHeapImplBase(const IsoHeapImplBase&) = delete;
    IsoHeapImplBase& operator=(const IsoHeapImplBase&) = delete;
public:
    static constexpr unsigned maxAllocationFromShared = 8;
    static constexpr unsigned maxAllocationFromSharedMask = (1U << maxAllocationFromShared) - 1U;
    static_assert(maxAllocationFromShared <= bmalloc::alignment);
    static_assert(isPowerOfTwo(maxAllocationFromShared));

    virtual ~IsoHeapImplBase();
    
    virtual void scavenge(Vector<DeferredDecommit>&) = 0;
    
    void scavengeNow();
    static void finishScavenging(Vector<DeferredDecommit>&);

    inline void didCommit(void* ptr, size_t bytes);
    inline void didDecommit(void* ptr, size_t bytes);

    inline void isNowFreeable(void* ptr, size_t bytes);
    inline void isNoLongerFreeable(void* ptr, size_t bytes);

    inline size_t freeableMemory();
    inline size_t footprint();

    void addToAllIsoHeaps();

protected:
    IsoHeapImplBase(Mutex&);

    friend class IsoSharedPage;
    friend class AllIsoHeaps;
    
public:
    // It's almost always the caller's responsibility to grab the lock. This lock comes from the
    // (*PerProcess<IsoTLSEntryHolder<IsoTLSDeallocatorEntry<Config>>>::get())->lock. That's pretty weird, and we don't
    // try to disguise the fact that it's weird. We only do that because heaps in the same size class
    // share the same deallocator log, so it makes sense for them to also share the same lock to
    // amortize lock acquisition costs.
    Mutex& lock;
protected:
    IsoHeapImplBase* m_next { nullptr };
    std::chrono::steady_clock::time_point m_lastSlowPathTime;
    size_t m_footprint { 0 };
    size_t m_freeableMemory { 0 };
#if ENABLE_PHYSICAL_PAGE_MAP
    PhysicalPageMap m_physicalPageMap;
#endif
    std::array<PackedAlignedPtr<uint8_t, bmalloc::alignment>, maxAllocationFromShared> m_sharedCells { };
protected:
    unsigned m_numberOfAllocationsFromSharedInOneCycle { 0 };
    unsigned m_availableShared { maxAllocationFromSharedMask };
    AllocationMode m_allocationMode { AllocationMode::Init };
    bool m_isInlineDirectoryEligibleOrDecommitted { true };
    static_assert(sizeof(m_availableShared) * 8 >= maxAllocationFromShared);
};

template<typename Config>
class IsoHeapImpl final : public IsoHeapImplBase {
    // Pick a size that makes us most efficiently use the bitvectors.
    static constexpr unsigned numPagesInInlineDirectory = 32;
    
public:
    IsoHeapImpl();
    
    EligibilityResult<Config> takeFirstEligible(const LockHolder&);
    
    // Callbacks from directory.
    void didBecomeEligibleOrDecommited(const LockHolder&, IsoDirectory<Config, numPagesInInlineDirectory>*);
    void didBecomeEligibleOrDecommited(const LockHolder&, IsoDirectory<Config, IsoDirectoryPage<Config>::numPages>*);
    
    void scavenge(Vector<DeferredDecommit>&) override;

    unsigned allocatorOffset();
    unsigned deallocatorOffset();

    // White-box testing functions.
    unsigned numLiveObjects();
    unsigned numCommittedPages();
    
    template<typename Func>
    void forEachDirectory(const LockHolder&, const Func&);
    
    template<typename Func>
    void forEachCommittedPage(const LockHolder&, const Func&);
    
    // This is only accurate when all threads are scavenged. Otherwise it will overestimate.
    template<typename Func>
    void forEachLiveObject(const LockHolder&, const Func&);

    AllocationMode updateAllocationMode();
    void* allocateFromShared(const LockHolder&, bool abortOnFailure);

private:
    PackedPtr<IsoDirectoryPage<Config>> m_headDirectory { nullptr };
    PackedPtr<IsoDirectoryPage<Config>> m_tailDirectory { nullptr };
    PackedPtr<IsoDirectoryPage<Config>> m_firstEligibleOrDecommitedDirectory { nullptr };
    IsoDirectory<Config, numPagesInInlineDirectory> m_inlineDirectory;
    unsigned m_nextDirectoryPageIndex { 1 }; // We start at 1 so that the high water mark being zero means we've only allocated in the inline directory since the last scavenge.
    unsigned m_directoryHighWatermark { 0 };
    IsoTLSEntryHolder<IsoTLSAllocatorEntry<Config>> m_allocator;
};

} // namespace bmalloc

#endif
#endif // !BUSE(TZONE)
