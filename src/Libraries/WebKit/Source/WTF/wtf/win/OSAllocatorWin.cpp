/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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

#include <windows.h>
#include <wtf/Assertions.h>
#include <wtf/DataLog.h>
#include <wtf/MathExtras.h>
#include <wtf/PageBlock.h>
#include <wtf/SoftLinking.h>

SOFT_LINK_LIBRARY(kernelbase)
SOFT_LINK_OPTIONAL(kernelbase, VirtualAlloc2, void*, WINAPI, (HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER *, ULONG))

namespace WTF {

static inline DWORD protection(bool writable, bool executable)
{
    return executable ?
        (writable ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ) :
        (writable ? PAGE_READWRITE : PAGE_READONLY);
}

void* OSAllocator::tryReserveUncommitted(size_t bytes, Usage, bool writable, bool executable, bool, bool)
{
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE, protection(writable, executable));
}

void* OSAllocator::reserveUncommitted(size_t bytes, Usage usage, bool writable, bool executable, bool jitCageEnabled, bool includesGuardPages)
{
    void* result = tryReserveUncommitted(bytes, usage, writable, executable, jitCageEnabled, includesGuardPages);
    RELEASE_ASSERT(result);
    return result;
}

void* OSAllocator::tryReserveUncommittedAligned(size_t bytes, size_t alignment, Usage, bool writable, bool executable, bool, bool)
{
    ASSERT(hasOneBitSet(alignment) && alignment >= pageSize());

    if (VirtualAlloc2Ptr()) {
        MEM_ADDRESS_REQUIREMENTS addressReqs = { };
        MEM_EXTENDED_PARAMETER param = { };
        addressReqs.Alignment = alignment;
        param.Type = MemExtendedParameterAddressRequirements;
        param.Pointer = &addressReqs;
        void* result = VirtualAlloc2Ptr()(nullptr, nullptr, bytes, MEM_RESERVE, protection(writable, executable), &param, 1);
        return result;
    }

    void* result = tryReserveUncommitted(bytes + alignment);
    // There's no way to release the reserved memory on Windows, from what I can tell as the whole segment has to be released at once.
    char* aligned = reinterpret_cast<char*>(roundUpToMultipleOf(alignment, reinterpret_cast<uintptr_t>(result)));
    return aligned;
}

void* OSAllocator::tryReserveAndCommit(size_t bytes, Usage, bool writable, bool executable, bool, bool)
{
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, protection(writable, executable));
}

void* OSAllocator::reserveAndCommit(size_t bytes, Usage usage, bool writable, bool executable, bool jitCageEnabled, bool includesGuardPages)
{
    void* result = tryReserveAndCommit(bytes, usage, writable, executable, jitCageEnabled, includesGuardPages);
    RELEASE_ASSERT(result);
    return result;
}

void OSAllocator::commit(void* address, size_t bytes, bool writable, bool executable)
{
    void* result = VirtualAlloc(address, bytes, MEM_COMMIT, protection(writable, executable));
    if (!result)
        CRASH();
}

void OSAllocator::decommit(void* address, size_t bytes)
{
    // https://docs.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc
    // Use MEM_RESET to purge physical pages at timing of OS's preference. This is aligned to
    // madvise MADV_FREE / MADV_FREE_REUSABLE.
    // https://devblogs.microsoft.com/oldnewthing/20170113-00/?p=95185
    // > The fact that MEM_RESET does not remove the page from the working set is not actually mentioned
    // > in the documentation for the MEM_RESET flag. Instead, itâ€™s mentioned in the documentation for
    // > the OfferÂ­VirtualÂ­Memory function, and in a sort of backhanded way
    // So, we need VirtualUnlock call.
    if (!bytes)
        return;
    void* result = VirtualAlloc(address, bytes, MEM_RESET, PAGE_READWRITE);
    if (!result)
        CRASH();
    // Calling VirtualUnlock on a range of memory that is not locked releases the pages from the
    // process's working set.
    // https://devblogs.microsoft.com/oldnewthing/20170317-00/?p=95755
    VirtualUnlock(address, bytes);
}

void OSAllocator::releaseDecommitted(void* address, size_t bytes)
{
    // See comment in OSAllocator::decommit(). Similarly, when bytes is 0, we
    // don't want to release anything. So, don't call VirtualFree() below.
    if (!bytes)
        return;
    // According to http://msdn.microsoft.com/en-us/library/aa366892(VS.85).aspx,
    // dwSize must be 0 if dwFreeType is MEM_RELEASE.
    bool result = VirtualFree(address, 0, MEM_RELEASE);
    if (!result)
        CRASH();
}

void OSAllocator::hintMemoryNotNeededSoon(void*, size_t)
{
}

bool OSAllocator::tryProtect(void* address, size_t bytes, bool readable, bool writable)
{
    if (!bytes)
        return true;
    DWORD protection = 0;
    if (readable) {
        if (writable)
            protection = PAGE_READWRITE;
        else
            protection = PAGE_READONLY;
    } else {
        ASSERT(!readable && !writable);
        protection = PAGE_NOACCESS;
    }
    return VirtualAlloc(address, bytes, MEM_COMMIT, protection);
}

void OSAllocator::protect(void* address, size_t bytes, bool readable, bool writable)
{
    if (bool result = tryProtect(address, bytes, readable, writable); UNLIKELY(!result)) {
        dataLogLn("mprotect failed: ", static_cast<int>(GetLastError()));
        RELEASE_ASSERT_NOT_REACHED();
    }
}

} // namespace WTF
