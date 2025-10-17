/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 17, 2023.
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
#ifndef PAS_SEGREGATED_PARTIAL_VIEW_INLINES_H
#define PAS_SEGREGATED_PARTIAL_VIEW_INLINES_H

#include "pas_bitvector.h"
#include "pas_segregated_partial_view.h"
#include "pas_shared_handle_or_page_boundary_inlines.h"

PAS_BEGIN_EXTERN_C;

typedef struct {
    size_t base_index;
    unsigned full_alloc_word;
    pas_segregated_shared_handle* handle;
    unsigned sharing_shift;
    pas_segregated_partial_view* view;
} pas_segregated_partial_view_tell_shared_handle_for_word_general_case_data;

static PAS_ALWAYS_INLINE unsigned
pas_segregated_partial_view_tell_shared_handle_for_word_general_case_source(
    size_t word_index, void* arg)
{
    pas_segregated_partial_view_tell_shared_handle_for_word_general_case_data*
        data;

    data = (pas_segregated_partial_view_tell_shared_handle_for_word_general_case_data*)arg;

    PAS_ASSERT(!word_index);

    return data->full_alloc_word;
}

static PAS_ALWAYS_INLINE bool
pas_segregated_partial_view_tell_shared_handle_for_word_general_case_callback(
    pas_found_bit_index index,
    void* arg)
{
    pas_segregated_partial_view_tell_shared_handle_for_word_general_case_data*
        data;
    size_t view_index;
    pas_compact_atomic_segregated_partial_view_ptr* partial_ptr;
    pas_segregated_partial_view* old_partial;

    data = (pas_segregated_partial_view_tell_shared_handle_for_word_general_case_data*)arg;

    view_index = (data->base_index + index.index) >> data->sharing_shift;
    partial_ptr = data->handle->partial_views + view_index;
    old_partial = pas_compact_atomic_segregated_partial_view_ptr_load(partial_ptr);
    PAS_ASSERT(!old_partial || old_partial == data->view);
    pas_compact_atomic_segregated_partial_view_ptr_store(partial_ptr, data->view);

    return true;
}

static PAS_ALWAYS_INLINE void
pas_segregated_partial_view_tell_shared_handle_for_word_general_case(
    pas_segregated_partial_view* view,
    pas_segregated_shared_handle* shared_handle,
    size_t word_index,
    unsigned full_alloc_word,
    unsigned sharing_shift)
{
    pas_segregated_partial_view_tell_shared_handle_for_word_general_case_data
        data;
    
    data.base_index = PAS_BITVECTOR_BIT_INDEX(word_index);
    data.full_alloc_word = full_alloc_word;
    data.handle = shared_handle;
    data.sharing_shift = sharing_shift;
    data.view = view;
    pas_bitvector_for_each_set_bit(
        pas_segregated_partial_view_tell_shared_handle_for_word_general_case_source,
        0, 1,
        pas_segregated_partial_view_tell_shared_handle_for_word_general_case_callback,
        &data);
}

static PAS_ALWAYS_INLINE void pas_segregated_partial_view_did_start_allocating(
    pas_segregated_partial_view* view,
    pas_segregated_shared_view* shared_view,
    pas_segregated_page_config page_config)
{
    pas_segregated_shared_handle* shared_handle;

    shared_handle = pas_unwrap_shared_handle(
        shared_view->shared_handle_or_page_boundary, page_config);

    pas_segregated_partial_view_set_is_in_use_for_allocation(view, shared_view, shared_handle);
}

PAS_END_EXTERN_C;

#endif /* PAS_SEGREGATED_PARTIAL_VIEW_INLINES_H */

