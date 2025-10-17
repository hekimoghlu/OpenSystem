/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 7, 2024.
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
#include "pas_heap.h"
#include "pas_segregated_heap_inlines.h"
#include "pas_segregated_view.h"

namespace {

void testMemalignArray(size_t size, size_t typeSize, size_t typeAlignment)
{
    bmalloc_type type = BMALLOC_TYPE_INITIALIZER(static_cast<unsigned>(typeSize),
                                                 static_cast<unsigned>(typeAlignment),
                                                 "test");
    pas_heap_ref heapRef = BMALLOC_HEAP_REF_INITIALIZER(&type);
    pas_segregated_view view;
    pas_segregated_size_directory* directory;

    void* ptr = bmalloc_iso_allocate_zeroed_array_by_size(&heapRef, size, pas_non_compact_allocation_mode);
    CHECK(ptr);

    view = pas_segregated_view_for_object(reinterpret_cast<uintptr_t>(ptr), &bmalloc_heap_config);
    directory = pas_segregated_view_get_size_directory(view);

    CHECK_EQUAL(pas_segregated_heap_size_directory_for_size(
                    &bmalloc_heap_ref_get_heap(&heapRef)->segregated_heap,
                    size,
                    BMALLOC_HEAP_CONFIG,
                    NULL),
                directory);
}

} // anonymous namespace

#endif // PAS_ENABLE_BMALLOC

void addMemalignTests()
{
#if PAS_ENABLE_BMALLOC
    ADD_TEST(testMemalignArray(1523, 16, 4));
    ADD_TEST(testMemalignArray(1523, 128, 128));
#endif // PAS_ENABLE_BMALLOC
}

