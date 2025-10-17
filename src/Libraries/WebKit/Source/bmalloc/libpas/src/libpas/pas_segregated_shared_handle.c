/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 25, 2024.
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

#include "pas_segregated_shared_handle.h"

#include "pas_heap_for_config.h"
#include "pas_segregated_shared_page_directory.h"
#include "pas_segregated_shared_view.h"
#include "pas_shared_handle_or_page_boundary_inlines.h"

pas_segregated_shared_handle* pas_segregated_shared_handle_create(
    pas_segregated_shared_view* view,
    pas_segregated_shared_page_directory* shared_page_directory,
    const pas_segregated_page_config* page_config_ptr)
{
    pas_segregated_page_config page_config;
    pas_segregated_shared_handle* result;
    size_t index;

    page_config = *page_config_ptr;

    result = pas_heap_for_page_config_allocate(
        page_config_ptr,
        pas_segregated_shared_handle_size(page_config),
        "pas_segregated_shared_handle");

    result->page_boundary = pas_unwrap_page_boundary(view->shared_handle_or_page_boundary);
    pas_compact_segregated_shared_view_ptr_store(&result->shared_view, view);
    result->directory = shared_page_directory;

    for (index = pas_segregated_shared_handle_num_views(page_config); index--;)
        pas_compact_atomic_segregated_partial_view_ptr_store(result->partial_views + index, NULL);

    view->shared_handle_or_page_boundary = pas_wrap_shared_handle(result, page_config);

    return result;
}

void pas_segregated_shared_handle_destroy(pas_segregated_shared_handle* handle)
{
    pas_segregated_shared_page_directory* shared_page_directory;
    pas_segregated_directory* directory;
    const pas_segregated_page_config* page_config_ptr;
    pas_segregated_page_config page_config;
    pas_segregated_shared_view* shared_view;

    shared_page_directory = handle->directory;
    directory = &shared_page_directory->base;
    page_config_ptr = pas_segregated_page_config_kind_get_config(directory->page_config_kind);
    page_config = *page_config_ptr;

    shared_view = pas_compact_segregated_shared_view_ptr_load(&handle->shared_view);

    PAS_ASSERT(pas_unwrap_shared_handle(shared_view->shared_handle_or_page_boundary,
                                        page_config) == handle);

    shared_view->shared_handle_or_page_boundary = pas_wrap_page_boundary(handle->page_boundary);

    pas_heap_for_page_config_deallocate(
        page_config_ptr,
        handle,
        pas_segregated_shared_handle_size(page_config));
}

void pas_segregated_shared_handle_note_emptiness(
    pas_segregated_shared_handle* handle)
{
    static const bool verbose = PAS_SHOULD_LOG(PAS_LOG_SEGREGATED_HEAPS);

    pas_segregated_shared_view* shared_view;

    shared_view = pas_compact_segregated_shared_view_ptr_load(&handle->shared_view);

    if (verbose) {
        pas_log("Marking handle %p, view %p empty, in_use_count = %u.\n",
                handle, shared_view, shared_view->is_in_use_for_allocation_count);
    }
    
    /* We currently don't want to notify emptiness until after we're done allocating in the
       page. However, this is a relatively weak requirement, since it's not the end of the
       world for a page to be empty but ineligible.
       
       Also, this will at worst happen once per granule. So this loop is likely not a
       terrible thing. */
    if (shared_view->is_in_use_for_allocation_count) {
        if (verbose)
            pas_log("Not setting shared %p as empty because it's in use.\n", shared_view);
        return;
    }

    if (verbose) {
        pas_log("Totally setting shared %p as empty because it's not in use.\n",
                shared_view);
    }
    
    pas_segregated_directory_view_did_become_empty(
        &handle->directory->base,
        pas_segregated_shared_view_as_view_non_null(shared_view));
}

#endif /* LIBPAS_ENABLED */
