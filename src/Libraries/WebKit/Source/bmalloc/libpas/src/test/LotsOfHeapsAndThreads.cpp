/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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

#if PAS_ENABLE_BMALLOC

#include "bmalloc_heap.h"
#include "bmalloc_heap_config.h"
#include "pas_get_heap.h"
#include <thread>

using namespace std;

namespace {

void testLotsOfHeapsAndThreads(unsigned numHeaps, unsigned numThreads, unsigned count)
{
    thread* threads = new thread[numThreads];
    pas_heap_ref* heaps = new pas_heap_ref[numHeaps];

    for (unsigned i = numHeaps; i--;)
        heaps[i] = BMALLOC_HEAP_REF_INITIALIZER(new bmalloc_type(BMALLOC_TYPE_INITIALIZER(16, 16, "test")));

    for (unsigned i = numThreads; i--;) {
        threads[i] = thread([&] () {
            for (unsigned j = count; j--;) {
                for (unsigned k = numHeaps; k--;) {
                    void* ptr = bmalloc_iso_allocate(heaps + k, pas_non_compact_allocation_mode);
                    CHECK_EQUAL(pas_get_heap(ptr, BMALLOC_HEAP_CONFIG),
                                bmalloc_heap_ref_get_heap(heaps + k));
                    bmalloc_deallocate(ptr);
                }
            }
        });
    }

    for (unsigned i = numThreads; i--;)
        threads[i].join();
}

} // anonymous namespace

#endif // PAS_ENABLE_BMALLOC

void addLotsOfHeapsAndThreadsTests()
{
#if PAS_ENABLE_BMALLOC
    ForceTLAs forceTLAs;
    ForcePartials forcePartials;
    
    ADD_TEST(testLotsOfHeapsAndThreads(10000, 100, 10));
    ADD_TEST(testLotsOfHeapsAndThreads(25000, 100, 10));
    ADD_TEST(testLotsOfHeapsAndThreads(30000, 100, 10)); // This is about as high as we can reliably go right now!
#endif // PAS_ENABLE_BMALLOC
}
