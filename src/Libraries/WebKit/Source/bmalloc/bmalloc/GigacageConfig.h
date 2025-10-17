/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
#include "GigacageKind.h"
#include <bit>
#include <inttypes.h>

namespace WebConfig {

using Slot = uint64_t;
extern "C" Slot g_config[];

} // namespace WebConfig

namespace Gigacage {

struct Config {
    void* basePtr(Kind kind) const
    {
        RELEASE_BASSERT(kind < NumberOfKinds);
        return basePtrs[static_cast<size_t>(kind)];
    }

    void setBasePtr(Kind kind, void* ptr)
    {
        RELEASE_BASSERT(kind < NumberOfKinds);
        basePtrs[static_cast<size_t>(kind)] = ptr;
    }

    void* allocBasePtr(Kind kind) const
    {
        RELEASE_BASSERT(kind < NumberOfKinds);
        return allocBasePtrs[static_cast<size_t>(kind)];
    }

    void setAllocBasePtr(Kind kind, void* ptr)
    {
        RELEASE_BASSERT(kind < NumberOfKinds);
        allocBasePtrs[static_cast<size_t>(kind)] = ptr;
    }

    size_t allocSize(Kind kind) const
    {
        RELEASE_BASSERT(kind < NumberOfKinds);
        return allocSizes[static_cast<size_t>(kind)];
    }

    void setAllocSize(Kind kind, size_t size)
    {
        RELEASE_BASSERT(kind < NumberOfKinds);
        allocSizes[static_cast<size_t>(kind)] = size;
    }

    // All the fields in this struct should be chosen such that their
    // initial value is 0 / null / falsy because Config is instantiated
    // as a global singleton.

    bool isPermanentlyFrozen; // Will be set by the client if the Config gets frozen.
    bool isEnabled;
    bool disablingPrimitiveGigacageIsForbidden;
    bool shouldBeEnabled;

    // We would like to just put the std::once_flag for these functions
    // here, but we can't because std::once_flag has a implicitly-deleted
    // default constructor. So, we use a boolean instead.
    bool shouldBeEnabledHasBeenCalled;
    bool ensureGigacageHasBeenCalled;

    void* start;
    size_t totalSize;
    void* basePtrs[static_cast<size_t>(NumberOfKinds)];
    void* allocBasePtrs[static_cast<size_t>(NumberOfKinds)];
    size_t allocSizes[static_cast<size_t>(NumberOfKinds)];
};

// The first 4 slots are reserved for the use of the ExecutableAllocator.
constexpr size_t startSlotOfGigacageConfig = 4;
constexpr size_t startOffsetOfGigacageConfig = startSlotOfGigacageConfig * sizeof(WebConfig::Slot);

constexpr size_t reservedSlotsForGigacageConfig = 16;
constexpr size_t reservedBytesForGigacageConfig = reservedSlotsForGigacageConfig * sizeof(WebConfig::Slot);

constexpr size_t alignmentOfGigacageConfig = std::alignment_of<Gigacage::Config>::value;

static_assert(sizeof(Gigacage::Config) + startOffsetOfGigacageConfig <= reservedBytesForGigacageConfig);
static_assert(bmalloc::roundUpToMultipleOf<alignmentOfGigacageConfig>(startOffsetOfGigacageConfig) == startOffsetOfGigacageConfig);

#define g_gigacageConfig (*std::bit_cast<Gigacage::Config*>(&WebConfig::g_config[Gigacage::startSlotOfGigacageConfig]))

} // namespace Gigacage
