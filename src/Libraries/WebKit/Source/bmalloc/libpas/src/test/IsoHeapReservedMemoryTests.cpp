/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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

#if PAS_ENABLE_ISO && SEGHEAP

#include "HeapLocker.h"
#include "iso_heap.h"

using namespace std;

namespace {

void testSizeProgression(size_t startSize,
                         size_t maxSize,
                         size_t sizeStep,
                         size_t countForSize,
                         size_t reservationSize,
                         bool shouldSucceed)
{
    static constexpr bool verbose = false;
    
    void* reservation = malloc(reservationSize);
    CHECK(reservation);
    
    pas_primitive_heap_ref heapRef = ISO_PRIMITIVE_HEAP_REF_INITIALIZER;

    uintptr_t begin = reinterpret_cast<uintptr_t>(reservation);
    uintptr_t end = begin + reservationSize;

    iso_force_primitive_heap_into_reserved_memory(&heapRef, begin, end);

    for (size_t size = startSize; size <= maxSize; size += sizeStep) {
        if (verbose)
            cout << "Allocating " << size << "\n";
        
        for (size_t i = countForSize; i--;) {
            void* object = iso_try_allocate_primitive(&heapRef, size, pas_non_compact_allocation_mode);
            
            if (shouldSucceed)
                CHECK(object);
            else {
                if (!object)
                    return;
            }
            
            uintptr_t objectBegin = reinterpret_cast<uintptr_t>(object);
            uintptr_t objectEnd = objectBegin + size;
            
            CHECK_GREATER_EQUAL(objectBegin, begin);
            CHECK_LESS(objectBegin, end);
            CHECK_GREATER_EQUAL(objectEnd, begin);
            CHECK_LESS_EQUAL(objectEnd, end);
        }
    }

    CHECK(shouldSucceed);
}

} // anonymous namespace

#endif // PAS_ENABLE_ISO && SEGHEAP

void addIsoHeapReservedMemoryTests()
{
#if PAS_ENABLE_ISO && SEGHEAP
    ADD_TEST(testSizeProgression(0, 1000000, 1024, 1, 1000000000, true));
    ADD_TEST(testSizeProgression(0, 10000, 64, 1000, 1000000000, true));
    ADD_TEST(testSizeProgression(0, 100000000, 1024, 1, 100000000, false));
#endif // PAS_ENABLE_ISO && SEGHEAP
}
