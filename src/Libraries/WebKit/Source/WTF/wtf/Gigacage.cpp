/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 5, 2024.
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
#include <wtf/Gigacage.h>

#include <wtf/Atomics.h>
#include <wtf/PageBlock.h>

#if USE(SYSTEM_MALLOC)
#include <wtf/OSAllocator.h>

namespace Gigacage {

void* tryMalloc(Kind, size_t size)
{
    return FastMalloc::tryMalloc(size);
}

void* tryZeroedMalloc(Kind, size_t size)
{
    return FastMalloc::tryZeroedMalloc(size);
}

void* tryRealloc(Kind, void* pointer, size_t size)
{
    return FastMalloc::tryRealloc(pointer, size);
}

void* tryAllocateZeroedVirtualPages(Kind, size_t requestedSize)
{
    size_t size = roundUpToMultipleOf(WTF::pageSize(), requestedSize);
    RELEASE_ASSERT(size >= requestedSize);
    void* result = OSAllocator::tryReserveAndCommit(size);
#if ASSERT_ENABLED
    if (result) {
        for (size_t i = 0; i < size / sizeof(uintptr_t); ++i)
            ASSERT(static_cast<uintptr_t*>(result)[i] == 0);
    }
#endif
    return result;
}

void freeVirtualPages(Kind, void* basePtr, size_t size)
{
    OSAllocator::decommitAndRelease(basePtr, size);
}

} // namespace Gigacage
#else // USE(SYSTEM_MALLOC)
#include <bmalloc/bmalloc.h>

namespace Gigacage {

// FIXME: Pointers into the primitive gigacage must be scrambled right after being returned from malloc,
// and stay scrambled except just before use.
// https://bugs.webkit.org/show_bug.cgi?id=175035

void* tryAlignedMalloc(Kind kind, size_t alignment, size_t size)
{
    void* result = bmalloc::api::tryMemalign(alignment, size, bmalloc::CompactAllocationMode::Compact, bmalloc::heapKind(kind));
    BPROFILE_TRY_ALLOCATION(GIGACAGE, kind, result, size);
    WTF::compilerFence();
    return result;
}

void alignedFree(Kind kind, void* p)
{
    if (!p)
        return;
    RELEASE_ASSERT(isCaged(kind, p));
    bmalloc::api::free(p, bmalloc::heapKind(kind));
    WTF::compilerFence();
}

void* tryMalloc(Kind kind, size_t size)
{
    void* result = bmalloc::api::tryMalloc(size, bmalloc::CompactAllocationMode::Compact, bmalloc::heapKind(kind));
    BPROFILE_TRY_ALLOCATION(GIGACAGE, kind, result, size);
    WTF::compilerFence();
    return result;
}

void* tryZeroedMalloc(Kind kind, size_t size)
{
    void* result = bmalloc::api::tryZeroedMalloc(size, bmalloc::CompactAllocationMode::Compact, bmalloc::heapKind(kind));
    BPROFILE_TRY_ALLOCATION(GIGACAGE, kind, result, size);
    WTF::compilerFence();
    return result;
}

void* tryRealloc(Kind kind, void* pointer, size_t size)
{
    void* result = bmalloc::api::tryRealloc(pointer, size, bmalloc::CompactAllocationMode::Compact, bmalloc::heapKind(kind));
    BPROFILE_TRY_ALLOCATION(GIGACAGE, kind, result, size);
    WTF::compilerFence();
    return result;
}

void free(Kind kind, void* p)
{
    if (!p)
        return;
    RELEASE_ASSERT(isCaged(kind, p));
    bmalloc::api::free(p, bmalloc::heapKind(kind));
    WTF::compilerFence();
}

void* tryAllocateZeroedVirtualPages(Kind kind, size_t size)
{
    void* result = bmalloc::api::tryLargeZeroedMemalignVirtual(WTF::pageSize(), size, bmalloc::CompactAllocationMode::Compact, bmalloc::heapKind(kind));
    BPROFILE_TRY_ALLOCATION(GIGACAGE, kind, result, size);
    WTF::compilerFence();
    return result;
}

void freeVirtualPages(Kind kind, void* basePtr, size_t size)
{
    if (!basePtr)
        return;
    RELEASE_ASSERT(isCaged(kind, basePtr));
    bmalloc::api::freeLargeVirtual(basePtr, size, bmalloc::heapKind(kind));
    WTF::compilerFence();
}

} // namespace Gigacage
#endif

namespace Gigacage {

void* tryMallocArray(Kind kind, size_t numElements, size_t elementSize)
{
    CheckedSize checkedSize = elementSize;
    checkedSize *= numElements;
    if (checkedSize.hasOverflowed())
        return nullptr;
    return tryMalloc(kind, checkedSize);
}

void* malloc(Kind kind, size_t size)
{
    void* result = tryMalloc(kind, size);
    RELEASE_ASSERT(result);
    return result;
}

void* zeroedMalloc(Kind kind, size_t size)
{
    void* result = tryZeroedMalloc(kind, size);
    RELEASE_ASSERT(result);
    return result;
}

void* mallocArray(Kind kind, size_t numElements, size_t elementSize)
{
    void* result = tryMallocArray(kind, numElements, elementSize);
    RELEASE_ASSERT(result);
    return result;
}

} // namespace Gigacage

