/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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

#include "pas_enumerate_bitfit_heaps.h"

#include "pas_bitfit_directory.h"
#include "pas_bitfit_heap.h"
#include "pas_bitfit_page.h"
#include "pas_bitfit_view.h"
#include "pas_enumerator_internal.h"

static bool view_callback(pas_enumerator* enumerator,
                          pas_compact_atomic_bitfit_view_ptr* view_ptr,
                          size_t index,
                          void* arg)
{
    static const bool verbose = PAS_SHOULD_LOG(PAS_LOG_BITFIT_HEAPS);
    
    pas_bitfit_view* view;
    pas_bitfit_directory* directory;
    const pas_bitfit_page_config* page_config;
    uintptr_t page_boundary;
    pas_bitfit_page* page;
    pas_page_granule_use_count* use_counts;
    uintptr_t page_size;
    uintptr_t granule_size;
    uintptr_t payload_begin;
    uintptr_t payload_end;
    uintptr_t offset;

    PAS_UNUSED_PARAM(index);
    
    directory = arg;

    view = pas_compact_atomic_bitfit_view_ptr_load_remote(enumerator, view_ptr);

    if (!view)
        return true;

    page_boundary = (uintptr_t)view->page_boundary;
    page_config = pas_bitfit_page_config_kind_get_config(directory->config_kind);
    
    if (page_boundary) {
        pas_enumerator_exclude_accounted_pages(
            enumerator, (void*)page_boundary, page_config->base.page_size);
    }

    if (!view->is_owned)
        return true;

    PAS_ASSERT_WITH_DETAIL(page_boundary);

    page = (pas_bitfit_page*)page_config->base.page_header_for_boundary_remote(
        enumerator, (void*)page_boundary);
    PAS_ASSERT_WITH_DETAIL(page);

    page = pas_enumerator_read(enumerator, page, pas_bitfit_page_header_size(*page_config));
    if (!page)
        return false;

    payload_begin = pas_bitfit_page_offset_to_first_object(*page_config);
    payload_end = pas_bitfit_page_offset_to_end_of_last_object(*page_config);

    page_size = page_config->base.page_size;
    granule_size = page_config->base.granule_size;

    if (page_size == granule_size)
        use_counts = NULL;
    else
        use_counts = pas_bitfit_page_get_granule_use_counts(page, *page_config);

    pas_enumerator_record_page_payload_and_meta(
        enumerator, page_boundary, page_size, granule_size, use_counts, payload_begin, payload_end);

    if (enumerator->record_object) {
        if (verbose)
            pas_log("Iterating objects in bitfit page %p\n", (void*)page_boundary);
        
        for (offset = payload_begin;
             offset < payload_end;
             offset += pas_page_base_config_min_align(page_config->base)) {
            uintptr_t second_offset;

            if (verbose)
                pas_log("offset = %lu\n", offset);
            
            if (pas_bitvector_get(pas_bitfit_page_free_bits(page),
                                  offset >> page_config->base.min_align_shift))
                continue;

            for (second_offset = offset;
                 second_offset < payload_end;
                 second_offset += pas_page_base_config_min_align(page_config->base)) {
                size_t second_index;

                second_index = second_offset >> page_config->base.min_align_shift;
                
                if (pas_bitvector_get(pas_bitfit_page_free_bits(page), second_index)) {
                    /* Found free bit before finding end bit; this means that this is an allocation
                       or deallocation in progress, so assume that it's a dead object. */
                    break;
                }
                
                if (pas_bitvector_get(pas_bitfit_page_object_end_bits(page, *page_config),
                                      second_index)) {
                    pas_enumerator_record(
                        enumerator,
                        (void*)(page_boundary + offset),
                        second_offset - offset + pas_page_base_config_min_align(page_config->base),
                        pas_enumerator_object_record);
                    break;
                }
            }

            offset = second_offset;
        }
    }
    
    return true;
}

static bool enumerate_bitfit_directory(pas_enumerator* enumerator,
                                       pas_bitfit_directory* directory)
{
    return pas_bitfit_directory_view_vector_iterate_remote(
        &directory->views, enumerator, 0, view_callback, directory);
}

static bool enumerate_bitfit_heap_callback(pas_enumerator* enumerator,
                                           pas_heap* heap,
                                           void* arg)
{
    pas_bitfit_heap* bitfit_heap;
    pas_bitfit_page_config_variant variant;
    
    PAS_ASSERT_WITH_DETAIL(!arg);

    bitfit_heap = pas_compact_atomic_bitfit_heap_ptr_load_remote(
        enumerator, &heap->segregated_heap.bitfit_heap);

    if (!bitfit_heap)
        return true;

    for (PAS_EACH_BITFIT_PAGE_CONFIG_VARIANT_ASCENDING(variant)) {
        if (!enumerate_bitfit_directory(
                enumerator, pas_bitfit_heap_get_directory(bitfit_heap, variant)))
            return false;
    }
    
    return true;
}

bool pas_enumerate_bitfit_heaps(pas_enumerator* enumerator)
{
    return pas_enumerator_for_each_heap(enumerator, enumerate_bitfit_heap_callback, NULL);
}

#endif /* LIBPAS_ENABLED */
