/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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

#include "Algorithm.h"
#include "BAssert.h"
#include "BExport.h"
#include "BInline.h"
#include "BPlatform.h"
#include "GigacageConfig.h"
#include "Sizes.h"
#include <bit>
#include <cstddef>
#include <inttypes.h>

#if BOS(DARWIN)
#include <mach/vm_param.h>
#endif

#if ((BOS(DARWIN) || BOS(LINUX)) && \
    (BCPU(X86_64) || (BCPU(ARM64) && !defined(__ILP32__) && (!BPLATFORM(IOS_FAMILY) || BPLATFORM(IOS)))))
#define GIGACAGE_ENABLED 1
#else
#define GIGACAGE_ENABLED 0
#endif


namespace Gigacage {

BINLINE const char* name(Kind kind)
{
    switch (kind) {
    case Primitive:
        return "Primitive";
    case NumberOfKinds:
        break;
    }
    BCRASH();
    return nullptr;
}

constexpr bool hasCapacityToUseLargeGigacage = BOS_EFFECTIVE_ADDRESS_WIDTH > 36;

#if GIGACAGE_ENABLED

constexpr size_t primitiveGigacageSize = (hasCapacityToUseLargeGigacage ? 64 : 16) * bmalloc::Sizes::GB;
constexpr size_t maximumCageSizeReductionForSlide = hasCapacityToUseLargeGigacage ? 4 * bmalloc::Sizes::GB : bmalloc::Sizes::GB / 4;


// In Linux, if `vm.overcommit_memory = 2` is specified, mmap with large size can fail if it exceeds the size of RAM.
// So we specify GIGACAGE_ALLOCATION_CAN_FAIL = 1.
#if BOS(LINUX)
#define GIGACAGE_ALLOCATION_CAN_FAIL 1
#else
#define GIGACAGE_ALLOCATION_CAN_FAIL 0
#endif


static_assert(bmalloc::isPowerOfTwo(primitiveGigacageSize));
static_assert(primitiveGigacageSize > maximumCageSizeReductionForSlide);

constexpr size_t gigacageSizeToMask(size_t size) { return size - 1; }

constexpr size_t primitiveGigacageMask = gigacageSizeToMask(primitiveGigacageSize);

// These constants are needed by the LLInt.
constexpr ptrdiff_t offsetOfPrimitiveGigacageBasePtr = static_cast<ptrdiff_t>(Primitive) * sizeof(void*);

extern "C" BEXPORT bool disablePrimitiveGigacageRequested;

BINLINE bool isEnabled() { return g_gigacageConfig.isEnabled; }

BEXPORT void ensureGigacage();

BEXPORT void disablePrimitiveGigacage();

// This will call the disable callback immediately if the Primitive Gigacage is currently disabled.
BEXPORT void addPrimitiveDisableCallback(void (*)(void*), void*);
BEXPORT void removePrimitiveDisableCallback(void (*)(void*), void*);

BEXPORT void forbidDisablingPrimitiveGigacage();

BINLINE bool disablingPrimitiveGigacageIsForbidden()
{
    return g_gigacageConfig.disablingPrimitiveGigacageIsForbidden;
}

BINLINE bool disableNotRequestedForPrimitiveGigacage()
{
    return !disablePrimitiveGigacageRequested;
}

BINLINE bool isEnabled(Kind kind)
{
    if (kind == Primitive)
        return g_gigacageConfig.basePtr(Primitive) && (disablingPrimitiveGigacageIsForbidden() || disableNotRequestedForPrimitiveGigacage());
    return g_gigacageConfig.basePtr(kind);
}

BINLINE void* basePtr(Kind kind)
{
    BASSERT(isEnabled(kind));
    return g_gigacageConfig.basePtr(kind);
}

BINLINE void* addressOfBasePtr(Kind kind)
{
    RELEASE_BASSERT(kind < NumberOfKinds);
    return &g_gigacageConfig.basePtrs[static_cast<size_t>(kind)];
}

BINLINE constexpr size_t maxSize(Kind kind)
{
    switch (kind) {
    case Primitive:
        return static_cast<size_t>(primitiveGigacageSize);
    case NumberOfKinds:
        break;
    }
    BCRASH();
    return 0;
}

BINLINE size_t alignment(Kind kind)
{
    return maxSize(kind);
}

BINLINE constexpr size_t mask(Kind kind)
{
    return gigacageSizeToMask(maxSize(kind));
}

BEXPORT void* allocBase(Kind);
BEXPORT size_t size(Kind);
BEXPORT size_t footprint(Kind);

template<typename Func>
void forEachKind(const Func& func)
{
    func(Primitive);
}

template<typename T>
BINLINE T* caged(Kind kind, T* ptr)
{
    BASSERT(ptr);
    if (!isEnabled(kind))
        return ptr;
    void* gigacageBasePtr = basePtr(kind);
    return reinterpret_cast<T*>(
        reinterpret_cast<uintptr_t>(gigacageBasePtr) + (
            reinterpret_cast<uintptr_t>(ptr) & mask(kind)));
}

template<typename T>
BINLINE T* cagedMayBeNull(Kind kind, T* ptr)
{
    if (!ptr)
        return ptr;
    return caged(kind, ptr);
}

BINLINE bool isCaged(Kind kind, const void* ptr)
{
    return caged(kind, ptr) == ptr;
}

BINLINE bool contains(const void* ptr)
{
    auto* start = reinterpret_cast<const uint8_t*>(g_gigacageConfig.start);
    auto* p = reinterpret_cast<const uint8_t*>(ptr);
    return static_cast<size_t>(p - start) < g_gigacageConfig.totalSize;
}

BEXPORT bool shouldBeEnabled();

#else // GIGACAGE_ENABLED

BINLINE void* basePtr(Kind)
{
    BCRASH();
    static void* unreachable;
    return unreachable;
}
BINLINE size_t maxSize(Kind) { BCRASH(); return 0; }
BINLINE size_t size(Kind) { return 0; }
BINLINE size_t footprint(Kind) { return 0; }
BINLINE void ensureGigacage() { }
BINLINE bool contains(const void*) { return false; }
BINLINE bool disablingPrimitiveGigacageIsForbidden() { return false; }
BINLINE bool isEnabled() { return false; }
BINLINE bool isCaged(Kind, const void*) { return true; }
BINLINE bool isEnabled(Kind) { return false; }
template<typename T> BINLINE T* caged(Kind, T* ptr) { return ptr; }
template<typename T> BINLINE T* cagedMayBeNull(Kind, T* ptr) { return ptr; }
BINLINE void forbidDisablingPrimitiveGigacage() { }
BINLINE void disablePrimitiveGigacage() { }
BINLINE void addPrimitiveDisableCallback(void (*)(void*), void*) { }
BINLINE void removePrimitiveDisableCallback(void (*)(void*), void*) { }

#endif // GIGACAGE_ENABLED

} // namespace Gigacage
