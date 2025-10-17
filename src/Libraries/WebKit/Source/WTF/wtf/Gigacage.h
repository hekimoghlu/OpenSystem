/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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

#include <wtf/FastMalloc.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/ASCIILiteral.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#if USE(SYSTEM_MALLOC)
#define GIGACAGE_ENABLED 0

namespace Gigacage {

constexpr bool hasCapacityToUseLargeGigacage = OS_CONSTANT(EFFECTIVE_ADDRESS_WIDTH) > 36;

const size_t primitiveGigacageMask = 0;

enum Kind {
    Primitive,
    NumberOfKinds
};

inline void ensureGigacage() { }
inline void disablePrimitiveGigacage() { }
inline bool shouldBeEnabled() { return false; }

inline void addPrimitiveDisableCallback(void (*)(void*), void*) { }
inline void removePrimitiveDisableCallback(void (*)(void*), void*) { }

inline void forbidDisablingPrimitiveGigacage() { }

ALWAYS_INLINE bool contains(const void*) { return false; }
ALWAYS_INLINE bool disablingPrimitiveGigacageIsForbidden() { return false; }
ALWAYS_INLINE bool isEnabled() { return false; }
ALWAYS_INLINE bool isEnabled(Kind) { return false; }
ALWAYS_INLINE size_t mask(Kind) { return 0; }
ALWAYS_INLINE size_t footprint(Kind) { return 0; }
ALWAYS_INLINE size_t maxSize(Kind) { return 0; }
ALWAYS_INLINE size_t size(Kind) { return 0; }

template<typename T>
inline T* caged(Kind, T* ptr) { return ptr; }
template<typename T>
inline T* cagedMayBeNull(Kind, T* ptr) { return ptr; }

inline bool isCaged(Kind, const void*) { return false; }

inline void* tryAlignedMalloc(Kind, size_t alignment, size_t size) { return tryFastAlignedMalloc(alignment, size); }
inline void alignedFree(Kind, void* p) { fastAlignedFree(p); }
WTF_EXPORT_PRIVATE void* tryMalloc(Kind, size_t size);
WTF_EXPORT_PRIVATE void* tryZeroedMalloc(Kind, size_t);
WTF_EXPORT_PRIVATE void* tryRealloc(Kind, void*, size_t);
inline void free(Kind, void* p) { fastFree(p); }

WTF_EXPORT_PRIVATE void* tryAllocateZeroedVirtualPages(Kind, size_t size);
WTF_EXPORT_PRIVATE void freeVirtualPages(Kind, void* basePtr, size_t size);

} // namespace Gigacage
#else
#include <bmalloc/Gigacage.h>

namespace Gigacage {

WTF_EXPORT_PRIVATE void* tryAlignedMalloc(Kind, size_t alignment, size_t size);
WTF_EXPORT_PRIVATE void alignedFree(Kind, void*);
WTF_EXPORT_PRIVATE void* tryMalloc(Kind, size_t);
WTF_EXPORT_PRIVATE void* tryZeroedMalloc(Kind, size_t);
WTF_EXPORT_PRIVATE void* tryRealloc(Kind, void*, size_t);
WTF_EXPORT_PRIVATE void free(Kind, void*);

WTF_EXPORT_PRIVATE void* tryAllocateZeroedVirtualPages(Kind, size_t size);
WTF_EXPORT_PRIVATE void freeVirtualPages(Kind, void* basePtr, size_t size);

} // namespace Gigacage
#endif

namespace Gigacage {

WTF_EXPORT_PRIVATE void* tryMallocArray(Kind, size_t numElements, size_t elementSize);

WTF_EXPORT_PRIVATE void* malloc(Kind, size_t);
WTF_EXPORT_PRIVATE void* zeroedMalloc(Kind, size_t);
WTF_EXPORT_PRIVATE void* mallocArray(Kind, size_t numElements, size_t elementSize);

} // namespace Gigacage

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
