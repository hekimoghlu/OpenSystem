/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 8, 2025.
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

#include "AllocationCounts.h"
#include "BAssert.h"
#include "BCompiler.h"
#include "BSyscall.h"
#include "BVMTags.h"
#include "Logging.h"
#include "Range.h"
#include "Sizes.h"
#include <algorithm>
#include <sys/mman.h>
#include <unistd.h>

#if BOS(DARWIN)
#include <mach/vm_page_size.h>
#endif

BALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace bmalloc {

#ifndef BMALLOC_VM_TAG
#define BMALLOC_VM_TAG VM_TAG_FOR_TCMALLOC_MEMORY
#endif

#if BOS(LINUX)
#define BMALLOC_NORESERVE MAP_NORESERVE
#else
#define BMALLOC_NORESERVE 0
#endif

inline size_t vmPageSize()
{
    static size_t cached;
    if (!cached) {
        long pageSize = sysconf(_SC_PAGESIZE);
        if (pageSize < 0)
            BCRASH();
        cached = pageSize;
    }
    return cached;
}

inline size_t vmPageShift()
{
    static size_t cached;
    if (!cached)
        cached = log2(vmPageSize());
    return cached;
}

inline size_t vmSize(size_t size)
{
    return roundUpToMultipleOf(vmPageSize(), size);
}

inline void vmValidate(size_t vmSize)
{
    BUNUSED(vmSize);
    BASSERT(vmSize);
    BASSERT(vmSize == roundUpToMultipleOf(vmPageSize(), vmSize));
}

inline void vmValidate(void* p, size_t vmSize)
{
    vmValidate(vmSize);
    
    BUNUSED(p);
    BASSERT(p);
    BASSERT(p == mask(p, ~(vmPageSize() - 1)));
}

inline size_t vmPageSizePhysical()
{
#if BOS(DARWIN) && (BCPU(ARM64) || BCPU(ARM))
    return vm_kernel_page_size;
#else
    static size_t cached;
    if (!cached)
        cached = sysconf(_SC_PAGESIZE);
    return cached;
#endif
}

inline void vmValidatePhysical(size_t vmSize)
{
    BUNUSED(vmSize);
    BASSERT(vmSize);
    BASSERT(vmSize == roundUpToMultipleOf(vmPageSizePhysical(), vmSize));
}

inline void vmValidatePhysical(void* p, size_t vmSize)
{
    vmValidatePhysical(vmSize);
    
    BUNUSED(p);
    BASSERT(p);
    BASSERT(p == mask(p, ~(vmPageSizePhysical() - 1)));
}

inline void* tryVMAllocate(size_t vmSize, VMTag usage = VMTag::Malloc)
{
    vmValidate(vmSize);

    BPROFILE_ALLOCATION(VM_ALLOCATION, vmSize, usage);

    void* result = mmap(0, vmSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | BMALLOC_NORESERVE, static_cast<int>(usage), 0);
    if (result == MAP_FAILED)
        return nullptr;
    return result;
}

inline void* vmAllocate(size_t vmSize, VMTag usage = VMTag::Malloc)
{
    void* result = tryVMAllocate(vmSize, usage);
    RELEASE_BASSERT(result);
    return result;
}

inline void vmDeallocate(void* p, size_t vmSize)
{
    vmValidate(p, vmSize);
    munmap(p, vmSize);
}

inline void vmRevokePermissions(void* p, size_t vmSize)
{
    vmValidate(p, vmSize);
    mprotect(p, vmSize, PROT_NONE);
}

inline void vmZeroAndPurge(void* p, size_t vmSize, VMTag usage = VMTag::Malloc)
{
    vmValidate(p, vmSize);
    int flags = MAP_PRIVATE | MAP_ANON | MAP_FIXED | BMALLOC_NORESERVE;
    int tag = static_cast<int>(usage);
    BPROFILE_ZERO_FILL_PAGE(p, vmSize, flags, tag);
    // MAP_ANON guarantees the memory is zeroed. This will also cause
    // page faults on accesses to this range following this call.
    void* result = mmap(p, vmSize, PROT_READ | PROT_WRITE, flags, tag, 0);
    RELEASE_BASSERT(result == p);
}

// Allocates vmSize bytes at a specified power-of-two alignment.
// Use this function to create maskable memory regions.

inline void* tryVMAllocate(size_t vmAlignment, size_t vmSize, VMTag usage = VMTag::Malloc)
{
    vmValidate(vmSize);
    vmValidate(vmAlignment);

    size_t mappedSize = vmAlignment + vmSize;
    if (mappedSize < vmAlignment || mappedSize < vmSize) // Check for overflow
        return nullptr;

    char* mapped = static_cast<char*>(tryVMAllocate(mappedSize, usage));
    if (!mapped)
        return nullptr;
    char* mappedEnd = mapped + mappedSize;

    char* aligned = roundUpToMultipleOf(vmAlignment, mapped);
    char* alignedEnd = aligned + vmSize;
    
    RELEASE_BASSERT(alignedEnd <= mappedEnd);
    
    if (size_t leftExtra = aligned - mapped)
        vmDeallocate(mapped, leftExtra);
    
    if (size_t rightExtra = mappedEnd - alignedEnd)
        vmDeallocate(alignedEnd, rightExtra);

    return aligned;
}

inline void* vmAllocate(size_t vmAlignment, size_t vmSize, VMTag usage = VMTag::Malloc)
{
    void* result = tryVMAllocate(vmAlignment, vmSize, usage);
    RELEASE_BASSERT(result);
    return result;
}

inline void vmDeallocatePhysicalPages(void* p, size_t vmSize)
{
    vmValidatePhysical(p, vmSize);
#if BOS(DARWIN)
    SYSCALL(madvise(p, vmSize, MADV_FREE_REUSABLE));
#elif BOS(FREEBSD)
    SYSCALL(madvise(p, vmSize, MADV_FREE));
#else
    SYSCALL(madvise(p, vmSize, MADV_DONTNEED));
#if BOS(LINUX)
    SYSCALL(madvise(p, vmSize, MADV_DONTDUMP));
#endif
#endif
}

inline void vmAllocatePhysicalPages(void* p, size_t vmSize)
{
    vmValidatePhysical(p, vmSize);
#if BOS(DARWIN)
    BUNUSED_PARAM(p);
    BUNUSED_PARAM(vmSize);
    // For the Darwin platform, we don't need to call madvise(..., MADV_FREE_REUSE)
    // to commit physical memory to back a range of allocated virtual memory.
    // Instead the kernel will commit pages as they are touched.
#else
    SYSCALL(madvise(p, vmSize, MADV_NORMAL));
#if BOS(LINUX)
    SYSCALL(madvise(p, vmSize, MADV_DODUMP));
#endif
#endif
}

// Returns how much memory you would commit/decommit had you called
// vmDeallocate/AllocatePhysicalPagesSloppy with p and size.
inline size_t physicalPageSizeSloppy(void* p, size_t size)
{
    char* begin = roundUpToMultipleOf(vmPageSizePhysical(), static_cast<char*>(p));
    char* end = roundDownToMultipleOf(vmPageSizePhysical(), static_cast<char*>(p) + size);

    if (begin >= end)
        return 0;
    return end - begin;
}

// Trims requests that are un-page-aligned.
inline void vmDeallocatePhysicalPagesSloppy(void* p, size_t size)
{
    char* begin = roundUpToMultipleOf(vmPageSizePhysical(), static_cast<char*>(p));
    char* end = roundDownToMultipleOf(vmPageSizePhysical(), static_cast<char*>(p) + size);

    if (begin >= end)
        return;

    vmDeallocatePhysicalPages(begin, end - begin);
}

// Expands requests that are un-page-aligned.
inline void vmAllocatePhysicalPagesSloppy(void* p, size_t size)
{
    char* begin = roundDownToMultipleOf(vmPageSizePhysical(), static_cast<char*>(p));
    char* end = roundUpToMultipleOf(vmPageSizePhysical(), static_cast<char*>(p) + size);

    if (begin >= end)
        return;

    vmAllocatePhysicalPages(begin, end - begin);
}

} // namespace bmalloc

BALLOW_UNSAFE_BUFFER_USAGE_END
