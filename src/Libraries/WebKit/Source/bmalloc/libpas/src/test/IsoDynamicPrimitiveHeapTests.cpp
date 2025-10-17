/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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

// This test is way too slow without segheaps.
#if PAS_ENABLE_ISO && SEGHEAP

#include <functional>
#include "iso_heap.h"
#include "iso_heap_innards.h"
#include "pas_dynamic_primitive_heap_map.h"
#include <set>
#include <vector>

using namespace std;

namespace {

void* allocate42(unsigned sizeIndex, const void* key)
{
    return iso_try_allocate_dynamic_primitive(key, 42 * (sizeIndex + 1), pas_non_compact_allocation_mode);
}

void* allocate42WithAlignment(unsigned sizeIndex, const void* key)
{
    return iso_try_allocate_dynamic_primitive_with_alignment(key, 42 * (sizeIndex + 1), 32, pas_non_compact_allocation_mode);
}

void* allocate42Zeroed(unsigned sizeIndex, const void* key)
{
    char* result = static_cast<char*>(
        iso_try_allocate_dynamic_primitive_zeroed(key, 42 * (sizeIndex + 1), pas_non_compact_allocation_mode));
    for (unsigned i = 42; i--;)
        CHECK(!result[i]);
    return result;
}

void* reallocate42(unsigned sizeIndex, const void* key)
{
    void* result = iso_try_allocate_common_primitive(16, pas_non_compact_allocation_mode);
    CHECK(result);
    return iso_try_reallocate_dynamic_primitive(
        result, key, 42 * (sizeIndex + 1), pas_reallocate_free_if_successful, pas_non_compact_allocation_mode);
}

void testManySizesAndKeys(
    unsigned numSizes, unsigned numKeys, unsigned maxHeaps,
    unsigned& maxHeapsLimiter,
    function<void*(unsigned sizeIndex, const void* key)> allocate)
{
    maxHeapsLimiter = maxHeaps;

    set<pas_heap*> heaps;
    vector<pas_heap*> heapByKey;

    for (unsigned keyIndex = 0; keyIndex < numKeys; ++keyIndex) {
        void* object = allocate(
            0,
            reinterpret_cast<const void*>(static_cast<uintptr_t>(keyIndex) + 666));
        CHECK(object);
        pas_heap* heap = iso_get_heap(object);
        CHECK(heap);
        CHECK_EQUAL(static_cast<bool>(heaps.count(heap)), keyIndex >= maxHeaps);
        heaps.insert(heap);
        PAS_ASSERT(heapByKey.size() == keyIndex);
        heapByKey.push_back(heap);
    }

    for (unsigned sizeIndex = 1; sizeIndex < numSizes; ++sizeIndex) {
        for (unsigned keyIndex = 0; keyIndex < numKeys; ++keyIndex) {
            void* object = allocate(
                0,
                reinterpret_cast<const void*>(static_cast<uintptr_t>(keyIndex) + 666));
            CHECK(object);
            pas_heap* heap = iso_get_heap(object);
            CHECK(heap);
            CHECK_EQUAL(heap, heapByKey[keyIndex]);
        }
    }
}

void testManySizesAndKeysInTandem(
    unsigned numSizesAndKeys, unsigned maxHeaps,
    unsigned& maxHeapsLimiter,
    bool disallowCollisions,
    function<void*(unsigned sizeIndex, const void* key)> allocate)
{
    iso_primitive_dynamic_heap_map.max_heaps_per_size = UINT_MAX;
    iso_primitive_dynamic_heap_map.max_heaps = UINT_MAX;
    
    maxHeapsLimiter = maxHeaps;

    set<pas_heap*> heaps;

    for (unsigned sizeAndKeyIndex = 0; sizeAndKeyIndex < numSizesAndKeys; ++sizeAndKeyIndex) {
        void* object = allocate(
            sizeAndKeyIndex,
            reinterpret_cast<const void*>(static_cast<uintptr_t>(sizeAndKeyIndex) + 666));
        CHECK(object);
        pas_heap* heap = iso_get_heap(object);
        CHECK(heap);
        if (disallowCollisions || sizeAndKeyIndex < maxHeaps) {
            CHECK(!heaps.count(heap));
            heaps.insert(heap);
        }
    }
}

} // anonymous namespace

#endif // PAS_ENABLE_ISO && SEGHEAP

void addIsoDynamicPrimitiveHeapTests()
{
#if PAS_ENABLE_ISO && SEGHEAP
    static constexpr bool skipTestsBecauseOfTLAExhaustionCrash = true;
    
    ADD_TEST(testManySizesAndKeys(100, 1, 10, iso_primitive_dynamic_heap_map.max_heaps_per_size, allocate42));
    ADD_TEST(testManySizesAndKeys(100, 1, 10, iso_primitive_dynamic_heap_map.max_heaps_per_size, allocate42WithAlignment));
    ADD_TEST(testManySizesAndKeys(100, 1, 10, iso_primitive_dynamic_heap_map.max_heaps_per_size, allocate42Zeroed));
    ADD_TEST(testManySizesAndKeys(100, 1, 10, iso_primitive_dynamic_heap_map.max_heaps_per_size, reallocate42));
    ADD_TEST(testManySizesAndKeys(100, 1, 1000, iso_primitive_dynamic_heap_map.max_heaps_per_size, allocate42));
    ADD_TEST(testManySizesAndKeys(100, 1, 1000, iso_primitive_dynamic_heap_map.max_heaps_per_size, allocate42WithAlignment));
    ADD_TEST(testManySizesAndKeys(100, 1, 1000, iso_primitive_dynamic_heap_map.max_heaps_per_size, allocate42Zeroed));
    ADD_TEST(testManySizesAndKeys(100, 1, 1000, iso_primitive_dynamic_heap_map.max_heaps_per_size, reallocate42));
    ADD_TEST(testManySizesAndKeys(100, 10000, 10, iso_primitive_dynamic_heap_map.max_heaps_per_size, allocate42));
    ADD_TEST(testManySizesAndKeys(100, 10000, 10, iso_primitive_dynamic_heap_map.max_heaps_per_size, allocate42WithAlignment));
    ADD_TEST(testManySizesAndKeys(100, 10000, 10, iso_primitive_dynamic_heap_map.max_heaps_per_size, allocate42Zeroed));
    ADD_TEST(testManySizesAndKeys(100, 10000, 10, iso_primitive_dynamic_heap_map.max_heaps_per_size, reallocate42));
    ADD_TEST(testManySizesAndKeys(100, 10000, 1000, iso_primitive_dynamic_heap_map.max_heaps_per_size, allocate42));
    ADD_TEST(testManySizesAndKeys(100, 10000, 1000, iso_primitive_dynamic_heap_map.max_heaps_per_size, allocate42WithAlignment));
    ADD_TEST(testManySizesAndKeys(100, 10000, 1000, iso_primitive_dynamic_heap_map.max_heaps_per_size, allocate42Zeroed));
    ADD_TEST(testManySizesAndKeys(100, 10000, 1000, iso_primitive_dynamic_heap_map.max_heaps_per_size, reallocate42));
    ADD_TEST(testManySizesAndKeysInTandem(100, 10, iso_primitive_dynamic_heap_map.max_heaps_per_size, true, allocate42));
    ADD_TEST(testManySizesAndKeysInTandem(100, 10, iso_primitive_dynamic_heap_map.max_heaps_per_size, true, allocate42WithAlignment));
    ADD_TEST(testManySizesAndKeysInTandem(100, 10, iso_primitive_dynamic_heap_map.max_heaps_per_size, true, allocate42Zeroed));
    ADD_TEST(testManySizesAndKeysInTandem(100, 10, iso_primitive_dynamic_heap_map.max_heaps_per_size, true, reallocate42));
    if (!skipTestsBecauseOfTLAExhaustionCrash) {
        ADD_TEST(testManySizesAndKeysInTandem(10000, 10, iso_primitive_dynamic_heap_map.max_heaps_per_size, true, allocate42));
        ADD_TEST(testManySizesAndKeysInTandem(10000, 10, iso_primitive_dynamic_heap_map.max_heaps_per_size, true, allocate42WithAlignment));
        ADD_TEST(testManySizesAndKeysInTandem(10000, 10, iso_primitive_dynamic_heap_map.max_heaps_per_size, true, allocate42Zeroed));
        ADD_TEST(testManySizesAndKeysInTandem(10000, 10, iso_primitive_dynamic_heap_map.max_heaps_per_size, true, reallocate42));
    }
    ADD_TEST(testManySizesAndKeysInTandem(100, 1000, iso_primitive_dynamic_heap_map.max_heaps_per_size, true, allocate42));
    ADD_TEST(testManySizesAndKeysInTandem(100, 1000, iso_primitive_dynamic_heap_map.max_heaps_per_size, true, allocate42WithAlignment));
    ADD_TEST(testManySizesAndKeysInTandem(100, 1000, iso_primitive_dynamic_heap_map.max_heaps_per_size, true, allocate42Zeroed));
    ADD_TEST(testManySizesAndKeysInTandem(100, 1000, iso_primitive_dynamic_heap_map.max_heaps_per_size, true, reallocate42));
    if (!skipTestsBecauseOfTLAExhaustionCrash) {
        ADD_TEST(testManySizesAndKeysInTandem(10000, 1000, iso_primitive_dynamic_heap_map.max_heaps_per_size, true, allocate42));
        ADD_TEST(testManySizesAndKeysInTandem(10000, 1000, iso_primitive_dynamic_heap_map.max_heaps_per_size, true, allocate42WithAlignment));
        ADD_TEST(testManySizesAndKeysInTandem(10000, 1000, iso_primitive_dynamic_heap_map.max_heaps_per_size, true, allocate42Zeroed));
        ADD_TEST(testManySizesAndKeysInTandem(10000, 1000, iso_primitive_dynamic_heap_map.max_heaps_per_size, true, reallocate42));
    }

    ADD_TEST(testManySizesAndKeys(100, 1, 10, iso_primitive_dynamic_heap_map.max_heaps, allocate42));
    ADD_TEST(testManySizesAndKeys(100, 1, 10, iso_primitive_dynamic_heap_map.max_heaps, allocate42WithAlignment));
    ADD_TEST(testManySizesAndKeys(100, 1, 10, iso_primitive_dynamic_heap_map.max_heaps, allocate42Zeroed));
    ADD_TEST(testManySizesAndKeys(100, 1, 10, iso_primitive_dynamic_heap_map.max_heaps, reallocate42));
    ADD_TEST(testManySizesAndKeys(100, 1, 1000, iso_primitive_dynamic_heap_map.max_heaps, allocate42));
    ADD_TEST(testManySizesAndKeys(100, 1, 1000, iso_primitive_dynamic_heap_map.max_heaps, allocate42WithAlignment));
    ADD_TEST(testManySizesAndKeys(100, 1, 1000, iso_primitive_dynamic_heap_map.max_heaps, allocate42Zeroed));
    ADD_TEST(testManySizesAndKeys(100, 1, 1000, iso_primitive_dynamic_heap_map.max_heaps, reallocate42));
    ADD_TEST(testManySizesAndKeys(100, 10000, 10, iso_primitive_dynamic_heap_map.max_heaps, allocate42));
    ADD_TEST(testManySizesAndKeys(100, 10000, 10, iso_primitive_dynamic_heap_map.max_heaps, allocate42WithAlignment));
    ADD_TEST(testManySizesAndKeys(100, 10000, 10, iso_primitive_dynamic_heap_map.max_heaps, allocate42Zeroed));
    ADD_TEST(testManySizesAndKeys(100, 10000, 10, iso_primitive_dynamic_heap_map.max_heaps, reallocate42));
    ADD_TEST(testManySizesAndKeys(100, 10000, 1000, iso_primitive_dynamic_heap_map.max_heaps, allocate42));
    ADD_TEST(testManySizesAndKeys(100, 10000, 1000, iso_primitive_dynamic_heap_map.max_heaps, allocate42WithAlignment));
    ADD_TEST(testManySizesAndKeys(100, 10000, 1000, iso_primitive_dynamic_heap_map.max_heaps, allocate42Zeroed));
    ADD_TEST(testManySizesAndKeys(100, 10000, 1000, iso_primitive_dynamic_heap_map.max_heaps, reallocate42));
    ADD_TEST(testManySizesAndKeysInTandem(100, 10, iso_primitive_dynamic_heap_map.max_heaps, false, allocate42));
    ADD_TEST(testManySizesAndKeysInTandem(100, 10, iso_primitive_dynamic_heap_map.max_heaps, false, allocate42WithAlignment));
    ADD_TEST(testManySizesAndKeysInTandem(100, 10, iso_primitive_dynamic_heap_map.max_heaps, false, allocate42Zeroed));
    ADD_TEST(testManySizesAndKeysInTandem(100, 10, iso_primitive_dynamic_heap_map.max_heaps, false, reallocate42));
    ADD_TEST(testManySizesAndKeysInTandem(10000, 10, iso_primitive_dynamic_heap_map.max_heaps, false, allocate42));
    ADD_TEST(testManySizesAndKeysInTandem(10000, 10, iso_primitive_dynamic_heap_map.max_heaps, false, allocate42WithAlignment));
    ADD_TEST(testManySizesAndKeysInTandem(10000, 10, iso_primitive_dynamic_heap_map.max_heaps, false, allocate42Zeroed));
    ADD_TEST(testManySizesAndKeysInTandem(10000, 10, iso_primitive_dynamic_heap_map.max_heaps, false, reallocate42));
    ADD_TEST(testManySizesAndKeysInTandem(100, 1000, iso_primitive_dynamic_heap_map.max_heaps, true, allocate42));
    ADD_TEST(testManySizesAndKeysInTandem(100, 1000, iso_primitive_dynamic_heap_map.max_heaps, true, allocate42WithAlignment));
    ADD_TEST(testManySizesAndKeysInTandem(100, 1000, iso_primitive_dynamic_heap_map.max_heaps, true, allocate42Zeroed));
    ADD_TEST(testManySizesAndKeysInTandem(100, 1000, iso_primitive_dynamic_heap_map.max_heaps, true, reallocate42));
    if (!skipTestsBecauseOfTLAExhaustionCrash) {
        ADD_TEST(testManySizesAndKeysInTandem(10000, 1000, iso_primitive_dynamic_heap_map.max_heaps, false, allocate42));
        ADD_TEST(testManySizesAndKeysInTandem(10000, 1000, iso_primitive_dynamic_heap_map.max_heaps, false, allocate42WithAlignment));
        ADD_TEST(testManySizesAndKeysInTandem(10000, 1000, iso_primitive_dynamic_heap_map.max_heaps, false, allocate42Zeroed));
        ADD_TEST(testManySizesAndKeysInTandem(10000, 1000, iso_primitive_dynamic_heap_map.max_heaps, false, reallocate42));
    }
#endif // PAS_ENABLE_ISO && SEGHEAP
}

