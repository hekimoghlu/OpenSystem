/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
#include "BPlatform.h"
#include "IsoMallocFallback.h"

#if !BUSE(TZONE)

#include "DebugHeap.h"
#include "Environment.h"
#include "bmalloc.h"

namespace bmalloc { namespace IsoMallocFallback {

MallocFallbackState mallocFallbackState;

namespace {

void determineMallocFallbackState()
{
    static std::once_flag onceFlag;
    std::call_once(
        onceFlag,
        [] {
            if (mallocFallbackState != MallocFallbackState::Undecided)
                return;

            if (Environment::get()->isDebugHeapEnabled()) {
                mallocFallbackState = MallocFallbackState::FallBackToMalloc;
                return;
            }

            const char* env = getenv("bmalloc_IsoHeap");
            if (env && (!strcasecmp(env, "false") || !strcasecmp(env, "no") || !strcmp(env, "0")))
                mallocFallbackState = MallocFallbackState::FallBackToMalloc;
            else
                mallocFallbackState = MallocFallbackState::DoNotFallBack;
        });
}

} // anonymous namespace

MallocResult tryMalloc(
    size_t size,
    [[maybe_unused]] CompactAllocationMode mode
#if BENABLE_MALLOC_HEAP_BREAKDOWN
    , malloc_zone_t* zone
#endif
    )
{
    for (;;) {
        switch (mallocFallbackState) {
        case MallocFallbackState::Undecided:
            determineMallocFallbackState();
            continue;
        case MallocFallbackState::FallBackToMalloc:
#if BENABLE_MALLOC_HEAP_BREAKDOWN
            return malloc_zone_malloc(zone, size);
#else
            return api::tryMalloc(size, mode);
#endif
        case MallocFallbackState::DoNotFallBack:
            return MallocResult();
        }
        RELEASE_BASSERT_NOT_REACHED();
    }
}

bool tryFree(
    void* ptr
#if BENABLE_MALLOC_HEAP_BREAKDOWN
    , malloc_zone_t* zone
#endif
    )
{
    for (;;) {
        switch (mallocFallbackState) {
        case MallocFallbackState::Undecided:
            determineMallocFallbackState();
            continue;
        case MallocFallbackState::FallBackToMalloc:
#if BENABLE_MALLOC_HEAP_BREAKDOWN
            malloc_zone_free(zone, ptr);
#else
            api::free(ptr);
#endif
            return true;
        case MallocFallbackState::DoNotFallBack:
            return false;
        }
        RELEASE_BASSERT_NOT_REACHED();
    }
}

} } // namespace bmalloc::IsoMallocFallback

#endif // !BUSE(TZONE)
