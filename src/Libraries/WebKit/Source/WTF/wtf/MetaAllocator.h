/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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

#include <wtf/Assertions.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/MetaAllocatorHandle.h>
#include <wtf/Noncopyable.h>
#include <wtf/PageBlock.h>
#include <wtf/RedBlackTree.h>
#include <wtf/RefPtr.h>

namespace WTF {

#define ENABLE_META_ALLOCATOR_PROFILE 0

class MetaAllocatorTracker {
    WTF_MAKE_FAST_ALLOCATED;
public:
    void notify(MetaAllocatorHandle&);
    void release(MetaAllocatorHandle&);

    MetaAllocatorHandle* find(void* address)
    {
        MetaAllocatorHandle* handle = m_allocations.findGreatestLessThanOrEqual(address);
        if (handle && handle->start().untaggedPtr() <= address && address < handle->end().untaggedPtr())
            return handle;
        return nullptr;
    }

    RedBlackTree<MetaAllocatorHandle, void*> m_allocations;
};

class MetaAllocator {
    WTF_MAKE_NONCOPYABLE(MetaAllocator);

public:
    using FreeSpacePtr = CodePtr<FreeSpacePtrTag>;
    using MemoryPtr = MetaAllocatorHandle::MemoryPtr;

    WTF_EXPORT_PRIVATE MetaAllocator(Lock&, size_t allocationGranule, size_t pageSize = WTF::pageSize());
    
    WTF_EXPORT_PRIVATE virtual ~MetaAllocator();
    
    ALWAYS_INLINE RefPtr<MetaAllocatorHandle> allocate(size_t sizeInBytes)
    {
        Locker locker { m_lock };
        return allocate(locker, sizeInBytes);
    }
    WTF_EXPORT_PRIVATE RefPtr<MetaAllocatorHandle> allocate(const Locker<Lock>&, size_t sizeInBytes);

    void trackAllocations(MetaAllocatorTracker* tracker)
    {
        m_tracker = tracker;
    }

    // Non-atomic methods for getting allocator statistics.
    size_t bytesAllocated() { return m_bytesAllocated; }
    size_t bytesReserved() { return m_bytesReserved; }
    size_t bytesCommitted() { return m_bytesCommitted; }
    
    // Atomic method for getting allocator statistics.
    struct Statistics {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        size_t bytesAllocated;
        size_t bytesReserved;
        size_t bytesCommitted;
    };
    Statistics currentStatistics()
    {
        Locker locker { m_lock };
        return currentStatistics(locker);
    }
    WTF_EXPORT_PRIVATE Statistics currentStatistics(const Locker<Lock>&);

    // Add more free space to the allocator. Call this directly from
    // the constructor if you wish to operate the allocator within a
    // fixed pool.
    WTF_EXPORT_PRIVATE void addFreshFreeSpace(void* start, size_t sizeInBytes);

    // This is meant only for implementing tests. Never call this in release
    // builds.
    WTF_EXPORT_PRIVATE size_t debugFreeSpaceSize();

    WTF_EXPORT_PRIVATE bool isInAllocatedMemory(const AbstractLocker&, void* address);
    
#if ENABLE(META_ALLOCATOR_PROFILE)
    void dumpProfile();
#else
    void dumpProfile() { }
#endif

protected:
    
    // Allocate new virtual space, but don't commit. This may return more
    // pages than we asked, in which case numPages is changed.
    virtual FreeSpacePtr allocateNewSpace(size_t& numPages) = 0;

    // Commit a page.
    virtual void notifyNeedPage(void* page, size_t) = 0;
    
    // Uncommit a page.
    virtual void notifyPageIsFree(void* page, size_t) = 0;
    
    // NOTE: none of the above methods are called during allocator
    // destruction, in part because a MetaAllocator cannot die so long
    // as there are Handles that refer to it.

    // Release a MetaAllocatorHandle.
    WTF_EXPORT_PRIVATE virtual void release(const Locker<Lock>&, MetaAllocatorHandle&);
private:
    
    friend class MetaAllocatorHandle;
    
    class FreeSpaceNode : public RedBlackTree<FreeSpaceNode, size_t>::Node {
    public:
        FreeSpaceNode() = default;

        size_t sizeInBytes()
        {
            return m_end.untaggedPtr<size_t>() - m_start.untaggedPtr<size_t>();
        }

        size_t key()
        {
            return sizeInBytes();
        }

        FreeSpacePtr m_start;
        FreeSpacePtr m_end;
    };
    typedef RedBlackTree<FreeSpaceNode, size_t> Tree;
    
    // Remove free space from the allocator. This is effectively
    // the allocate() function, except that it does not mark the
    // returned space as being in-use.
    FreeSpacePtr findAndRemoveFreeSpace(size_t sizeInBytes);

    // This is called when memory from an allocation is freed.
    void addFreeSpaceFromReleasedHandle(FreeSpacePtr start, size_t sizeInBytes);

    // This is the low-level implementation of adding free space; it
    // is called from both addFreeSpaceFromReleasedHandle and from
    // addFreshFreeSpace.
    void addFreeSpace(FreeSpacePtr start, size_t sizeInBytes);

    // Management of used space.
    
    void incrementPageOccupancy(void* address, size_t sizeInBytes);
    void decrementPageOccupancy(void* address, size_t sizeInBytes);

    // Utilities.
    
    size_t roundUp(size_t sizeInBytes);
    
    FreeSpaceNode* allocFreeSpaceNode();
    WTF_EXPORT_PRIVATE void freeFreeSpaceNode(FreeSpaceNode*);
    
    size_t m_allocationGranule;
    size_t m_pageSize;
    unsigned m_logAllocationGranule;
    unsigned m_logPageSize;
    
    Tree m_freeSpaceSizeMap;
    UncheckedKeyHashMap<FreeSpacePtr, FreeSpaceNode*> m_freeSpaceStartAddressMap;
    UncheckedKeyHashMap<FreeSpacePtr, FreeSpaceNode*> m_freeSpaceEndAddressMap;
    UncheckedKeyHashMap<uintptr_t, size_t> m_pageOccupancyMap;
    
    size_t m_bytesAllocated;
    size_t m_bytesReserved;
    size_t m_bytesCommitted;
    
    Lock& m_lock;

    MetaAllocatorTracker* m_tracker { nullptr };

#ifndef NDEBUG
    size_t m_mallocBalance;
#endif

#if ENABLE(META_ALLOCATOR_PROFILE)
    unsigned m_numAllocations;
    unsigned m_numFrees;
#endif
};

} // namespace WTF
