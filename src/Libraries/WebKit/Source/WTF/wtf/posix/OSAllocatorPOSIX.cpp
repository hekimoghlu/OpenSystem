/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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
#include "config.h"
#include <wtf/OSAllocator.h>

#include <errno.h>
#include <sys/mman.h>
#include <wtf/Assertions.h>
#include <wtf/DataLog.h>
#include <wtf/MallocSpan.h>
#include <wtf/MathExtras.h>
#include <wtf/Mmap.h>
#include <wtf/PageBlock.h>
#include <wtf/SafeStrerror.h>
#include <wtf/text/CString.h>

#if ENABLE(JIT_CAGE)
#include <WebKitAdditions/JITCageAdditions.h>
#else // ENABLE(JIT_CAGE)
#if OS(DARWIN)
#define MAP_EXECUTABLE_FOR_JIT MAP_JIT
#define MAP_EXECUTABLE_FOR_JIT_WITH_JIT_CAGE MAP_JIT
#else // OS(DARWIN)
#define MAP_EXECUTABLE_FOR_JIT 0
#define MAP_EXECUTABLE_FOR_JIT_WITH_JIT_CAGE 0
#endif // OS(DARWIN)
#endif // ENABLE(JIT_CAGE)

#if PLATFORM(COCOA)
#include <wtf/spi/cocoa/MachVMSPI.h>
#endif

namespace WTF {

void* OSAllocator::tryReserveAndCommit(size_t bytes, Usage usage, bool writable, bool executable, bool jitCageEnabled, bool includesGuardPages)
{
    // All POSIX reservations start out logically committed.
    int protection = PROT_READ;
    if (writable)
        protection |= PROT_WRITE;
    if (executable)
        protection |= PROT_EXEC;

    int flags = MAP_PRIVATE | MAP_ANON;
#if OS(DARWIN)
    if (executable) {
        if (jitCageEnabled)
            flags |= MAP_EXECUTABLE_FOR_JIT_WITH_JIT_CAGE;
        else
            flags |= MAP_EXECUTABLE_FOR_JIT;
    }
#else
    UNUSED_PARAM(jitCageEnabled);
#endif

#if OS(DARWIN)
    int fd = usage;
#else
    UNUSED_PARAM(usage);
    int fd = -1;
#endif

    auto result = MallocSpan<uint8_t, Mmap>::mmap(bytes, protection, flags, fd);
    if (result && includesGuardPages) {
        // We use mmap to remap the guardpages rather than using mprotect as
        // mprotect results in multiple references to the code region. This
        // breaks the madvise based mechanism we use to return physical memory
        // to the OS.
        mmap(result.mutableSpan().data(), pageSize(), PROT_NONE, MAP_FIXED | MAP_PRIVATE | MAP_ANON, fd, 0);
        mmap(result.mutableSpan().last(pageSize()).data(), pageSize(), PROT_NONE, MAP_FIXED | MAP_PRIVATE | MAP_ANON, fd, 0);
    }
    return result.leakSpan().data();
}

void* OSAllocator::tryReserveUncommitted(size_t bytes, Usage usage, bool writable, bool executable, bool jitCageEnabled, bool includesGuardPages)
{
#if OS(LINUX) || OS(HAIKU)
    UNUSED_PARAM(usage);
    UNUSED_PARAM(jitCageEnabled);
    UNUSED_PARAM(includesGuardPages);
    int protection = PROT_READ;
    if (writable)
        protection |= PROT_WRITE;
    if (executable)
        protection |= PROT_EXEC;

    void* result = mmap(0, bytes, protection, MAP_NORESERVE | MAP_PRIVATE | MAP_ANON, -1, 0);
    if (result == MAP_FAILED)
        result = nullptr;
    if (result)
        while (madvise(result, bytes, MADV_DONTNEED) == -1 && errno == EAGAIN) { }
#else
    void* result = tryReserveAndCommit(bytes, usage, writable, executable, jitCageEnabled, includesGuardPages);
#if HAVE(MADV_FREE_REUSE)
    if (result) {
        // To support the "reserve then commit" model, we have to initially decommit.
        while (madvise(result, bytes, MADV_FREE_REUSABLE) == -1 && errno == EAGAIN) { }
    }
#endif

#endif

    return result;
}

void* OSAllocator::reserveUncommitted(size_t bytes, Usage usage, bool writable, bool executable, bool jitCageEnabled, bool includesGuardPages)
{
    void* result = tryReserveUncommitted(bytes, usage, writable, executable, jitCageEnabled, includesGuardPages);
    RELEASE_ASSERT(result);
    return result;
}

void* OSAllocator::tryReserveUncommittedAligned(size_t bytes, size_t alignment, Usage usage, bool writable, bool executable, bool jitCageEnabled, bool includesGuardPages)
{
    ASSERT(hasOneBitSet(alignment) && alignment >= pageSize());

#if PLATFORM(MAC) || USE(APPLE_INTERNAL_SDK)
    UNUSED_PARAM(usage); // Not supported for mach API.
    ASSERT_UNUSED(includesGuardPages, !includesGuardPages);
    ASSERT_UNUSED(jitCageEnabled, !jitCageEnabled); // Not supported for mach API.
    vm_prot_t protections = VM_PROT_READ;
    if (writable)
        protections |= VM_PROT_WRITE;
    if (executable)
        protections |= VM_PROT_EXECUTE;

    const vm_inherit_t childProcessInheritance = VM_INHERIT_DEFAULT;
    const bool copy = false;
    const int flags = VM_FLAGS_ANYWHERE;

    void* aligned = nullptr;
    kern_return_t result = mach_vm_map(mach_task_self(), reinterpret_cast<mach_vm_address_t*>(&aligned), bytes, alignment - 1, flags, MEMORY_OBJECT_NULL, 0, copy, protections, protections, childProcessInheritance);
    ASSERT_UNUSED(result, result == KERN_SUCCESS || !aligned);
#if HAVE(MADV_FREE_REUSE)
    if (aligned) {
        // To support the "reserve then commit" model, we have to initially decommit.
        while (madvise(aligned, bytes, MADV_FREE_REUSABLE) == -1 && errno == EAGAIN) { }
    }
#endif

    return aligned;
#else
#if HAVE(MAP_ALIGNED)
#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif
    UNUSED_PARAM(usage);
    UNUSED_PARAM(jitCageEnabled);
    UNUSED_PARAM(includesGuardPages);
    int protection = PROT_READ;
    if (writable)
        protection |= PROT_WRITE;
    if (executable)
        protection |= PROT_EXEC;

    void* result = mmap(0, bytes, protection, MAP_NORESERVE | MAP_PRIVATE | MAP_ANON | MAP_ALIGNED(getLSBSet(alignment)), -1, 0);
    if (result == MAP_FAILED)
        return nullptr;
    if (result)
        while (madvise(result, bytes, MADV_DONTNEED) == -1 && errno == EAGAIN) { }
    return result;
#else

    // Add the alignment so we can ensure enough mapped memory to get an aligned start.
    size_t mappedSize = bytes + alignment;
    auto* rawMapped = reinterpret_cast<uint8_t*>(tryReserveUncommitted(mappedSize, usage, writable, executable, jitCageEnabled, includesGuardPages));
    if (!rawMapped)
        return nullptr;
    auto mappedSpan = unsafeMakeSpan(rawMapped, mappedSize);

    auto* rawAligned = reinterpret_cast<uint8_t*>(roundUpToMultipleOf(alignment, reinterpret_cast<uintptr_t>(mappedSpan.data())));
    auto alignedSpan = mappedSpan.subspan(rawAligned - mappedSpan.data(), bytes);

    if (size_t leftExtra = alignedSpan.data() - mappedSpan.data())
        releaseDecommitted(mappedSpan.data(), leftExtra);

    if (size_t rightExtra = std::to_address(mappedSpan.end()) - std::to_address(alignedSpan.end()))
        releaseDecommitted(std::to_address(alignedSpan.end()), rightExtra);

    return alignedSpan.data();
#endif // HAVE(MAP_ALIGNED)
#endif // PLATFORM(MAC) || USE(APPLE_INTERNAL_SDK)
}

void* OSAllocator::reserveAndCommit(size_t bytes, Usage usage, bool writable, bool executable, bool jitCageEnabled, bool includesGuardPages)
{
    void* result = tryReserveAndCommit(bytes, usage, writable, executable, jitCageEnabled, includesGuardPages);
    RELEASE_ASSERT(result);
    return result;
}

void OSAllocator::commit(void* address, size_t bytes, bool writable, bool executable)
{
#if OS(LINUX) || OS(HAIKU)
    UNUSED_PARAM(writable);
    UNUSED_PARAM(executable);
    while (madvise(address, bytes, MADV_WILLNEED) == -1 && errno == EAGAIN) { }
#elif HAVE(MADV_FREE_REUSE)
    UNUSED_PARAM(writable);
    UNUSED_PARAM(executable);
    while (madvise(address, bytes, MADV_FREE_REUSE) == -1 && errno == EAGAIN) { }
#else
    // Non-MADV_FREE_REUSE reservations automatically commit on demand.
    UNUSED_PARAM(address);
    UNUSED_PARAM(bytes);
    UNUSED_PARAM(writable);
    UNUSED_PARAM(executable);
#endif
}

void OSAllocator::decommit(void* address, size_t bytes)
{
#if OS(LINUX) || OS(HAIKU)
    while (madvise(address, bytes, MADV_DONTNEED) == -1 && errno == EAGAIN) { }
#elif HAVE(MADV_FREE_REUSE)
    while (madvise(address, bytes, MADV_FREE_REUSABLE) == -1 && errno == EAGAIN) { }
#elif HAVE(MADV_FREE)
    while (madvise(address, bytes, MADV_FREE) == -1 && errno == EAGAIN) { }
#elif HAVE(MADV_DONTNEED)
    while (madvise(address, bytes, MADV_DONTNEED) == -1 && errno == EAGAIN) { }
#else
    UNUSED_PARAM(address);
    UNUSED_PARAM(bytes);
#endif
}

void OSAllocator::hintMemoryNotNeededSoon(void* address, size_t bytes)
{
#if HAVE(MADV_DONTNEED)
    while (madvise(address, bytes, MADV_DONTNEED) == -1 && errno == EAGAIN) { }
#else
    UNUSED_PARAM(address);
    UNUSED_PARAM(bytes);
#endif
}

void OSAllocator::releaseDecommitted(void* address, size_t bytes)
{
    int result = munmap(address, bytes);
    if (result == -1)
        CRASH();
}

bool OSAllocator::tryProtect(void* address, size_t bytes, bool readable, bool writable)
{
    int protection = 0;
    if (readable) {
        if (writable)
            protection = PROT_READ | PROT_WRITE;
        else
            protection = PROT_READ;
    } else {
        ASSERT(!readable && !writable);
        protection = PROT_NONE;
    }
    return !mprotect(address, bytes, protection);
}

void OSAllocator::protect(void* address, size_t bytes, bool readable, bool writable)
{
    if (bool result = tryProtect(address, bytes, readable, writable); UNLIKELY(!result)) {
        dataLogLn("mprotect failed: ", safeStrerror(errno).data());
        RELEASE_ASSERT_NOT_REACHED();
    }
}

} // namespace WTF
