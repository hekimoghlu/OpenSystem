/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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
#include "pas_segregated_size_directory.h"
#include <set>

using namespace std;

namespace {

void setupConfig()
{
    // If these assertions ever fail we could just fix it by replacing them with code that mutates the
    // config to have the settings we want.
    CHECK_EQUAL(bmalloc_intrinsic_runtime_config.base.view_cache_capacity_for_object_size,
                pas_heap_runtime_config_aggressive_view_cache_capacity);
    CHECK_EQUAL(bmalloc_intrinsic_runtime_config.base.directory_size_bound_for_no_view_cache, 0);
}

void testDisableViewCacheUsingBoundForNoViewCache()
{
    setupConfig();

    bmalloc_intrinsic_runtime_config.base.directory_size_bound_for_no_view_cache = UINT_MAX;

    bmalloc_deallocate(bmalloc_allocate(42, pas_non_compact_allocation_mode));
}

void testEnableViewCacheAtSomeBoundForNoViewCache(unsigned bound)
{
    setupConfig();

    bmalloc_intrinsic_runtime_config.base.directory_size_bound_for_no_view_cache = bound;

    void* ptr = bmalloc_allocate(42, pas_non_compact_allocation_mode);
    pas_segregated_view view = pas_segregated_view_for_object(
        reinterpret_cast<uintptr_t>(ptr), &bmalloc_heap_config);
    pas_segregated_size_directory* theDirectory = pas_segregated_view_get_size_directory(view);

    CHECK_EQUAL(theDirectory->view_cache_index, UINT_MAX);

    set<pas_segregated_view> views;
    views.insert(view);

    for (;;) {
        ptr = bmalloc_allocate(42, pas_non_compact_allocation_mode);
        view = pas_segregated_view_for_object(reinterpret_cast<uintptr_t>(ptr), &bmalloc_heap_config);
        pas_segregated_size_directory* directory = pas_segregated_view_get_size_directory(view);

        CHECK_EQUAL(directory, theDirectory);

        views.insert(view);

        if (views.size() > bound) {
            CHECK_EQUAL(views.size(), bound + 1);
            CHECK_LESS(theDirectory->view_cache_index, UINT_MAX);
            CHECK_GREATER(theDirectory->view_cache_index, 0);

            pas_segregated_page* page = pas_segregated_view_get_page(view);
            CHECK_EQUAL(page->view_cache_index, theDirectory->view_cache_index);
            return;
        }
    }
}

} // anonymous namespace

#endif // PAS_ENABLE_BMALLOC

void addViewCacheTests()
{
#if PAS_ENABLE_BMALLOC
    ForceExclusives forceExclusives;
    ForceTLAs forceTLAs;
    
    ADD_TEST(testDisableViewCacheUsingBoundForNoViewCache());
    ADD_TEST(testEnableViewCacheAtSomeBoundForNoViewCache(1));
    ADD_TEST(testEnableViewCacheAtSomeBoundForNoViewCache(10));
    ADD_TEST(testEnableViewCacheAtSomeBoundForNoViewCache(100));
#endif // PAS_ENABLE_BMALLOC
}

