/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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

#include <algorithm>
#include <wtf/VMTags.h>

namespace WTF {

class OSAllocator {
    WTF_MAKE_FAST_ALLOCATED;
public:
    enum Usage {
        UnknownUsage = -1,
        FastMallocPages = VM_TAG_FOR_TCMALLOC_MEMORY,
        JSJITCodePages = VM_TAG_FOR_EXECUTABLEALLOCATOR_MEMORY,
    };

    // The requested alignment must be a power of two and greater than the system page size.
    // The memory returned by this cannot be released as on Windows there's no guaranteed API to
    // get an aligned address and the size + alignment then rounding trick cannot release the unused parts
    // due to how the Windows syscalls work.
    WTF_EXPORT_PRIVATE static void* tryReserveUncommittedAligned(size_t size, size_t alignment, Usage = UnknownUsage, bool writable = true, bool executable = false, bool jitCageEnabled = false, bool includesGuardPages = false);

    // These methods are symmetric; reserveUncommitted allocates VM in an uncommitted state,
    // releaseDecommitted should be called on a region of VM allocated by a single reservation,
    // the memory must all currently be in a decommitted state. reserveUncommitted returns to
    // you memory that is zeroed.
    WTF_EXPORT_PRIVATE static void* reserveUncommitted(size_t, Usage = UnknownUsage, bool writable = true, bool executable = false, bool jitCageEnabled = false, bool includesGuardPages = false);
    WTF_EXPORT_PRIVATE static void* tryReserveUncommitted(size_t, Usage = UnknownUsage, bool writable = true, bool executable = false, bool jitCageEnabled = false, bool includesGuardPages = false);
    WTF_EXPORT_PRIVATE static void releaseDecommitted(void*, size_t);

    // These methods are symmetric; they commit or decommit a region of VM (uncommitted VM should
    // never be accessed, since the OS may not have attached physical memory for these regions).
    // Clients should only call commit on uncommitted regions and decommit on committed regions.
    WTF_EXPORT_PRIVATE static void commit(void*, size_t, bool writable, bool executable);
    WTF_EXPORT_PRIVATE static void decommit(void*, size_t);

    // These methods are symmetric; reserveAndCommit allocates VM in an committed state,
    // decommitAndRelease should be called on a region of VM allocated by a single reservation,
    // the memory must all currently be in a committed state.
    WTF_EXPORT_PRIVATE static void* reserveAndCommit(size_t, Usage = UnknownUsage, bool writable = true, bool executable = false, bool jitCageEnabled = false, bool includesGuardPages = false);
    WTF_EXPORT_PRIVATE static void* tryReserveAndCommit(size_t, Usage = UnknownUsage, bool writable = true, bool executable = false, bool jitCageEnabled = false, bool includesGuardPages = false);
    static void decommitAndRelease(void* base, size_t size);

    // These methods are akin to reserveAndCommit/decommitAndRelease, above - however rather than
    // committing/decommitting the entire region additional parameters allow a subregion to be
    // specified.
    WTF_EXPORT_PRIVATE static void* reserveAndCommit(size_t reserveSize, size_t commitSize, Usage = UnknownUsage, bool writable = true, bool executable = false, bool jitCageEnabled = false);

    // Reallocate an existing, committed allocation.
    // The prior allocation must be fully comitted, and the new size will also be fully committed.
    // This interface is provided since it may be possible to optimize this operation on some platforms.
    template<typename T>
    static T* reallocateCommitted(T*, size_t oldSize, size_t newSize, Usage = UnknownUsage, bool writable = true, bool executable = false, bool jitCageEnabled = false);

    // Hint to the OS that an address range is not expected to be accessed anytime soon.
    WTF_EXPORT_PRIVATE static void hintMemoryNotNeededSoon(void*, size_t);

    WTF_EXPORT_PRIVATE static void protect(void*, size_t, bool readable, bool writable);
    WTF_EXPORT_PRIVATE static bool tryProtect(void*, size_t, bool readable, bool writable);
};

inline void* OSAllocator::reserveAndCommit(size_t reserveSize, size_t commitSize, Usage usage, bool writable, bool executable, bool jitCageEnabled)
{
    void* base = reserveUncommitted(reserveSize, usage, writable, executable, jitCageEnabled);
    commit(base, commitSize, writable, executable);
    return base;
}

inline void OSAllocator::decommitAndRelease(void* releaseBase, size_t releaseSize)
{
    releaseDecommitted(releaseBase, releaseSize);
}

template<typename T>
inline T* OSAllocator::reallocateCommitted(T* oldBase, size_t oldSize, size_t newSize, Usage usage, bool writable, bool executable, bool jitCageEnabled)
{
    void* newBase = reserveAndCommit(newSize, usage, writable, executable, jitCageEnabled);
    memcpy(newBase, oldBase, std::min(oldSize, newSize));
    decommitAndRelease(oldBase, oldSize);
    return static_cast<T*>(newBase);
}

} // namespace WTF

using WTF::OSAllocator;
