/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 11, 2025.
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
#include "TestHarness.h"
#include "bmalloc_heap.h"
#include "bmalloc_heap_config.h"

#include <cstdlib>

using namespace std;

namespace {

void testBmallocAllocate()
{
    void* mem = bmalloc_try_allocate(100, pas_non_compact_allocation_mode);
    CHECK(mem);
}

void testBmallocDeallocate()
{
    void* mem = bmalloc_try_allocate(100, pas_non_compact_allocation_mode);
    CHECK(mem);
    bmalloc_deallocate(mem);
}

void testBmallocForceBitfitAfterAlloc()
{
    void* mem0 = bmalloc_try_allocate(28616, pas_non_compact_allocation_mode);
    CHECK(mem0);

    void* mem1 = bmalloc_try_allocate(20768, pas_non_compact_allocation_mode);
    CHECK(mem1);

    // Simulate entering mini mode by forcing bitfit only.
    bmalloc_intrinsic_runtime_config.base.max_segregated_object_size = 0;
    bmalloc_intrinsic_runtime_config.base.max_bitfit_object_size = UINT_MAX;
    bmalloc_primitive_runtime_config.base.max_segregated_object_size = 0;
    bmalloc_primitive_runtime_config.base.max_bitfit_object_size = UINT_MAX;

    void* mem2 = bmalloc_try_allocate(20648, pas_non_compact_allocation_mode);
    CHECK(mem2);
}

void testBmallocSmallIndexOverlap()
{
    // object_size = 16 * index for this heap.
    // Creates directory A with min_index = 97, object_size = 1616
    void* mem0 = bmalloc_try_allocate(1552, pas_non_compact_allocation_mode);
    CHECK(mem0);
    // Extends directory A to have min_index = 96, object_size = 1616
    void* mem1 = bmalloc_try_allocate(1536, pas_non_compact_allocation_mode);
    CHECK(mem1);
    // Install index is 94. Directory A is a "candidate" but doesn't satisfy alignment,
    // so new directory B is created with min_index = 94, object_size = 1536.
    // Directory B overlaps directory A at index 96 (1536 / 16).
    void* mem2 = bmalloc_try_allocate_with_alignment(1504, 32, pas_non_compact_allocation_mode);
    CHECK(mem2);
}

} // anonymous namespace

void addBmallocTests()
{
    ADD_TEST(testBmallocAllocate());
    ADD_TEST(testBmallocDeallocate());
    ADD_TEST(testBmallocForceBitfitAfterAlloc());
    ADD_TEST(testBmallocSmallIndexOverlap());
}
