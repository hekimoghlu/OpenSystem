/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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

#if PAS_ENABLE_JIT

#include <iostream>
#include "jit_heap.h"
#include "jit_heap_config.h"
#include "pas_utils.h"

using namespace std;

namespace {

void testAllocateShrinkAndAllocate(unsigned initialObjectSize,
                                   unsigned numInitialObjects,
                                   unsigned initialSizeOfObjectToShrink,
                                   unsigned sizeToShrinkTo,
                                   unsigned finalObjectSize,
                                   unsigned expectedAlignmentOfFinalObject)
{
    for (unsigned i = numInitialObjects; i--;) {
        void* ptr = jit_heap_try_allocate(initialObjectSize);
        CHECK(ptr);
    }

    PAS_ASSERT(sizeToShrinkTo <= initialSizeOfObjectToShrink);

    void* ptr = jit_heap_try_allocate(initialSizeOfObjectToShrink);
    CHECK(ptr);

    jit_heap_shrink(ptr, sizeToShrinkTo);

    uintptr_t expectedAddress = pas_round_up_to_power_of_2(
        reinterpret_cast<uintptr_t>(ptr) + (sizeToShrinkTo ? sizeToShrinkTo : 1),
        expectedAlignmentOfFinalObject);

    void* newPtr = jit_heap_try_allocate(finalObjectSize);

    CHECK_EQUAL(
        newPtr,
        reinterpret_cast<void*>(expectedAddress));

    jit_heap_deallocate(ptr);
    jit_heap_deallocate(newPtr);
}

void testAllocationSize(size_t requestedSize, size_t actualSize)
{
    CHECK_EQUAL(jit_heap_get_size(jit_heap_try_allocate(requestedSize)), actualSize);
}

} // anonymous namespace

#endif // PAS_ENABLE_JIT

void addJITHeapTests()
{
#if PAS_ENABLE_JIT
    BootJITHeap bootJITHeap;

    {
        ForceBitfit forceBitfit;
        ADD_TEST(testAllocateShrinkAndAllocate(0, 0, 0, 0, 0, 4));
        ADD_TEST(testAllocateShrinkAndAllocate(0, 0, 128, 64, 64, 4));
        ADD_TEST(testAllocateShrinkAndAllocate(32, 10, 128, 64, 64, 4));
        ADD_TEST(testAllocateShrinkAndAllocate(32, 10, 1000, 500, 1000, 4));
        ADD_TEST(testAllocateShrinkAndAllocate(0, 0, 2048, 512, 1100, 256));
        ADD_TEST(testAllocateShrinkAndAllocate(32, 10, 2048, 512, 1100, 256));
        ADD_TEST(testAllocateShrinkAndAllocate(1100, 10, 2048, 512, 1100, 256));
        ADD_TEST(testAllocateShrinkAndAllocate(0, 0, 100000, 10000, 80000, 4));
    }
    
    ADD_TEST(testAllocationSize(4, 16));
    ADD_TEST(testAllocationSize(8, 16));
    ADD_TEST(testAllocationSize(12, 16));
    ADD_TEST(testAllocationSize(16, 16));
    ADD_TEST(testAllocationSize(20, 32));
    {
        ForceBitfit forceBitfit;
        ADD_TEST(testAllocationSize(4, 4));
        ADD_TEST(testAllocationSize(8, 8));
        ADD_TEST(testAllocationSize(12, 12));
        ADD_TEST(testAllocationSize(16, 16));
        ADD_TEST(testAllocationSize(20, 20));
    }
#endif // PAS_ENABLE_JIT
}
