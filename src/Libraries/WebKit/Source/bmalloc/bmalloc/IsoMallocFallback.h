/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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

#include "BExport.h"
#include "BInline.h"
#include "BPlatform.h"
#include "CompactAllocationMode.h"
#include <cstddef>
#include <cstdint>

#if BENABLE_MALLOC_HEAP_BREAKDOWN
#include <malloc/malloc.h>
#endif

namespace bmalloc { namespace IsoMallocFallback {

enum class MallocFallbackState : uint8_t {
    Undecided,
    FallBackToMalloc,
    DoNotFallBack
};

extern MallocFallbackState mallocFallbackState;

BINLINE bool shouldTryToFallBack()
{
    if ((true))
        return false;
    
    return mallocFallbackState != MallocFallbackState::DoNotFallBack;
}
    
struct MallocResult {
    MallocResult() = default;
    
    MallocResult(void* ptr)
        : ptr(ptr)
        , didFallBack(true)
    {
    }
    
    void* ptr { nullptr } ;
    bool didFallBack { false };
};

BEXPORT MallocResult tryMalloc(
    size_t size,
    CompactAllocationMode mode
#if BENABLE_MALLOC_HEAP_BREAKDOWN
    , malloc_zone_t* zone = nullptr
#endif
    );

// Returns true if it did fall back.
BEXPORT bool tryFree(
    void* ptr
#if BENABLE_MALLOC_HEAP_BREAKDOWN
    , malloc_zone_t* zone = nullptr
#endif
    );
    
} } // namespace bmalloc::IsoMallocFallback

#endif // !BUSE(TZONE)
