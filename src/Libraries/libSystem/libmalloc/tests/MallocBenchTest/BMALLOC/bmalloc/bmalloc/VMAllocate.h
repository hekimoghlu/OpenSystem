/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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
#ifndef VMAllocate_h
#define VMAllocate_h

#include "BAssert.h"
#include "Logging.h"
#include "Range.h"
#include "Sizes.h"
#include "Syscall.h"
#include <algorithm>
#include <sys/mman.h>
#include <unistd.h>

#if BOS(DARWIN)
#include <mach/vm_page_size.h>
#include <mach/vm_statistics.h>
#endif

namespace bmalloc {

#if BOS(DARWIN)
#define BMALLOC_VM_TAG VM_MAKE_TAG(VM_MEMORY_TCMALLOC)
#define BMALLOC_NORESERVE 0
#elif BOS(LINUX)
#define BMALLOC_VM_TAG -1
#define BMALLOC_NORESERVE MAP_NORESERVE
#else
#define BMALLOC_VM_TAG -1
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
#if BPLATFORM(IOS_FAMILY)
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

inline void* tryVMAllocate(size_t vmSize)
{
    vmValidate(vmSize);
    void* result = mmap(0, vmSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | BMALLOC_NORESERVE, BMALLOC_VM_TAG, 0);
    if (result == MAP_FAILED)
        return nullptr;
    return result;
}

inline void* vmAllocate(size_t vmSize)
{
    void* result = tryVMAllocate(vmSize);
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

inline void vmZeroAndPurge(void* p, size_t vmSize)
{
    vmValidate(p, vmSize);
    // MAP_ANON guarantees the memory is zeroed. This will also cause
    // page faults on accesses to this range following this call.
    void* result = mmap(p, vmSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED | BMALLOC_NORESERVE, BMALLOC_VM_TAG, 0);
    RELEASE_BASSERT(result == p);
}

// Allocates vmSize bytes at a specified power-of-two alignment.
// Use this function to create maskable memory regions.

inline void* tryVMAllocate(size_t vmAlignment, size_t vmSize)
{
    vmValidate(vmSize);
    vmValidate(vmAlignment);

    size_t mappedSize = vmAlignment + vmSize;
    if (mappedSize < vmAlignment || mappedSize < vmSize) // Check for overflow
        return nullptr;

    char* mapped = static_cast<char*>(tryVMAllocate(mappedSize));
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

inline void* vmAllocate(size_t vmAlignment, size_t vmSize)
{
    void* result = tryVMAllocate(vmAlignment, vmSize);
    RELEASE_BASSERT(result);
    return result;
}

inline void vmDeallocatePhysicalPages(void* p, size_t vmSize)
{
    vmValidatePhysical(p, vmSize);
#if BOS(DARWIN)
    SYSCALL(madvise(p, vmSize, MADV_FREE_REUSABLE));
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
    SYSCALL(madvise(p, vmSize, MADV_FREE_REUSE));
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

#endif // VMAllocate_h
