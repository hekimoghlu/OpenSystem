/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
#ifndef PAS_FULL_ALLOC_BITS_INLINES_H
#define PAS_FULL_ALLOC_BITS_INLINES_H

#include "pas_full_alloc_bits.h"
#include "pas_segregated_exclusive_view.h"
#include "pas_segregated_page_config.h"
#include "pas_segregated_partial_view.h"
#include "pas_segregated_size_directory.h"
#include "pas_segregated_view.h"

PAS_BEGIN_EXTERN_C;

static PAS_ALWAYS_INLINE pas_full_alloc_bits
pas_full_alloc_bits_create_for_exclusive(
    pas_segregated_size_directory* directory,
    pas_segregated_page_config page_config)
{
    return pas_full_alloc_bits_create(
        pas_compact_tagged_unsigned_ptr_load_non_null(
            &pas_segregated_size_directory_data_ptr_load_non_null(
                &directory->data)->full_alloc_bits),
        0,
        (unsigned)pas_segregated_page_config_num_alloc_words(page_config));
}

static PAS_ALWAYS_INLINE pas_full_alloc_bits
pas_full_alloc_bits_create_for_partial_but_not_primordial(pas_segregated_view view)
{
    pas_segregated_partial_view* partial_view;
    
    partial_view = pas_segregated_view_get_partial(view);

    return pas_full_alloc_bits_create(
        pas_lenient_compact_unsigned_ptr_load_compact_non_null(&partial_view->alloc_bits),
        partial_view->alloc_bits_offset,
        partial_view->alloc_bits_offset + partial_view->alloc_bits_size);
}

static PAS_ALWAYS_INLINE pas_full_alloc_bits
pas_full_alloc_bits_create_for_partial(pas_segregated_view view)
{
    pas_segregated_partial_view* partial_view;
    
    partial_view = pas_segregated_view_get_partial(view);

    return pas_full_alloc_bits_create(
        pas_lenient_compact_unsigned_ptr_load(&partial_view->alloc_bits),
        partial_view->alloc_bits_offset,
        partial_view->alloc_bits_offset + partial_view->alloc_bits_size);
}

static PAS_ALWAYS_INLINE pas_full_alloc_bits
pas_full_alloc_bits_create_for_view_and_directory(
    pas_segregated_view view,
    pas_segregated_size_directory* directory,
    pas_segregated_page_config page_config)
{
    if (pas_segregated_view_is_some_exclusive(view))
        return pas_full_alloc_bits_create_for_exclusive(directory, page_config);

    return pas_full_alloc_bits_create_for_partial(view);
}

static PAS_ALWAYS_INLINE pas_full_alloc_bits
pas_full_alloc_bits_create_for_view(
    pas_segregated_view view,
    pas_segregated_page_config page_config)
{
    if (pas_segregated_view_is_some_exclusive(view)) {
        pas_segregated_size_directory* size_directory;
        size_directory = pas_compact_segregated_size_directory_ptr_load_non_null(
            &pas_segregated_view_get_exclusive(view)->directory);
        return pas_full_alloc_bits_create_for_exclusive(size_directory, page_config);
    }

    return pas_full_alloc_bits_create_for_partial(view);
}

PAS_END_EXTERN_C;

#endif /* PAS_FULL_ALLOC_BITS_INLINES_H */

