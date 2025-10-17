/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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

#if PAS_ENABLE_ISO

#include "iso_heap.h"

namespace {

void testIsoAllocate(size_t requestedSize, size_t actualSize)
{
    pas_heap_ref heap = ISO_HEAP_REF_INITIALIZER(requestedSize);

    void* object = iso_allocate(&heap, pas_non_compact_allocation_mode);
    CHECK_EQUAL(iso_get_allocation_size(object), actualSize);
    CHECK_LESS(heap.allocator_index, 10000);

    void* object2 = iso_allocate(&heap, pas_non_compact_allocation_mode);
    CHECK_NOT_EQUAL(object, object2);
    CHECK_EQUAL(iso_get_allocation_size(object2), actualSize);
}

void testAllocatePrimitive(size_t requestedSize, size_t actualSize)
{
    pas_primitive_heap_ref heap = ISO_PRIMITIVE_HEAP_REF_INITIALIZER;

    void* object = iso_allocate_primitive(&heap, requestedSize, pas_non_compact_allocation_mode);
    CHECK_EQUAL(iso_get_allocation_size(object), actualSize);
    CHECK_LESS(heap.base.allocator_index, 10000);

    void* object2 = iso_allocate_primitive(&heap, requestedSize, pas_non_compact_allocation_mode);
    CHECK_NOT_EQUAL(object, object2);
    CHECK_EQUAL(iso_get_allocation_size(object2), actualSize);
}

} // anonymous namespace

#endif // PAS_ENABLE_ISO

void addHeapRefAllocatorIndexTests()
{
#if PAS_ENABLE_ISO
    ADD_TEST(testIsoAllocate(16, 16));
    ADD_TEST(testIsoAllocate(32, 32));
    ADD_TEST(testIsoAllocate(48, 48));
    ADD_TEST(testIsoAllocate(64, 64));
    ADD_TEST(testIsoAllocate(1000, 1008));
    ADD_TEST(testIsoAllocate(1008, 1008));
    ADD_TEST(testIsoAllocate(10000, 10752));
    ADD_TEST(testIsoAllocate(10752, 10752));
    
    ADD_TEST(testAllocatePrimitive(16, 16));
    ADD_TEST(testAllocatePrimitive(32, 32));
    ADD_TEST(testAllocatePrimitive(48, 48));
    ADD_TEST(testAllocatePrimitive(64, 64));
    ADD_TEST(testAllocatePrimitive(1000, 1008));
    ADD_TEST(testAllocatePrimitive(1008, 1008));
    ADD_TEST(testAllocatePrimitive(10000, 10752));
    ADD_TEST(testAllocatePrimitive(10752, 10752));
#endif // PAS_ENABLE_ISO
}

