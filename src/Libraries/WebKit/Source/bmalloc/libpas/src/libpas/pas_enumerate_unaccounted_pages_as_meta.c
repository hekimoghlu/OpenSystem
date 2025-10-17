/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 21, 2021.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_enumerate_unaccounted_pages_as_meta.h"

#include "pas_enumerator.h"
#include "pas_ptr_hash_set.h"
#include "pas_ptr_min_heap.h"
#include "pas_root.h"

bool pas_enumerate_unaccounted_pages_as_meta(pas_enumerator* enumerator)
{
    size_t index;
    pas_ptr_min_heap ptr_heap;
    void* span_begin;
    void* span_end;
    void* page;

    if (!enumerator->record_meta)
        return true;

    pas_ptr_min_heap_construct(&ptr_heap);

    for (index = pas_ptr_hash_set_entry_index_end(enumerator->unaccounted_pages); index--;) {
        page = *pas_ptr_hash_set_entry_at_index(enumerator->unaccounted_pages, index);

        if (pas_ptr_hash_set_entry_is_empty_or_deleted(page))
            continue;

        PAS_ASSERT_WITH_DETAIL(page);

        pas_ptr_min_heap_add(&ptr_heap, page, &enumerator->allocation_config);
    }

    span_begin = NULL;
    span_end = NULL;
    while ((page = pas_ptr_min_heap_take_min(&ptr_heap))) {
        if (span_end != page) {
            PAS_ASSERT_WITH_DETAIL(page > span_end);
            pas_enumerator_record(
                enumerator,
                span_begin,
                (uintptr_t)span_end - (uintptr_t)span_begin,
                pas_enumerator_meta_record);
            span_begin = page;
        }
        span_end = (char*)page + enumerator->root->page_malloc_alignment;
    }
    pas_enumerator_record(
        enumerator,
        span_begin,
        (uintptr_t)span_end - (uintptr_t)span_begin,
        pas_enumerator_meta_record);

    return true;
}

#endif /* LIBPAS_ENABLED */
