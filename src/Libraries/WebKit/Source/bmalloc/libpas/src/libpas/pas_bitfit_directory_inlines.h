/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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
#ifndef PAS_BITFIT_DIRECTORY_INLINES_H
#define PAS_BITFIT_DIRECTORY_INLINES_H

#include "pas_bitfit_directory.h"
#include "pas_bitfit_page_config.h"

PAS_BEGIN_EXTERN_C;

typedef struct {
    unsigned num_bits;
} pas_bitfit_directory_find_first_free_for_num_bits_iterate_data;

static PAS_ALWAYS_INLINE bool
pas_bitfit_directory_find_first_free_for_num_bits_iterate_callback(
    pas_bitfit_max_free* entry_ptr,
    size_t index,
    void* arg)
{
    pas_bitfit_directory_find_first_free_for_num_bits_iterate_data* data;
    pas_bitfit_max_free entry;

    PAS_UNUSED_PARAM(index);

    data = arg;

    entry = *entry_ptr;

    /* Return true if we want to keep going. */
    if (entry < data->num_bits)
        return true;
    
    if (entry == PAS_BITFIT_MAX_FREE_EMPTY)
        return true;
    
    return false;
}

static PAS_ALWAYS_INLINE pas_found_index
pas_bitfit_directory_find_first_free_for_num_bits(pas_bitfit_directory* directory,
                                                  unsigned start_index,
                                                  unsigned num_bits)
{
    pas_bitfit_directory_find_first_free_for_num_bits_iterate_data data;
    
    PAS_TESTING_ASSERT(num_bits <= (unsigned)UINT8_MAX);

    data.num_bits = num_bits;

    return pas_bitfit_directory_max_free_vector_iterate(
        &directory->max_frees, start_index,
        pas_bitfit_directory_find_first_free_for_num_bits_iterate_callback, &data);
}

static PAS_ALWAYS_INLINE pas_found_index
pas_bitfit_directory_find_first_free(pas_bitfit_directory* directory,
                                     unsigned start_index,
                                     unsigned size,
                                     pas_bitfit_page_config page_config)
{
    return pas_bitfit_directory_find_first_free_for_num_bits(
        directory, start_index, size >> page_config.base.min_align_shift);
}

static PAS_ALWAYS_INLINE bool
pas_bitfit_directory_find_first_empty_iterate_callback(
    pas_bitfit_max_free* entry_ptr,
    size_t index,
    void* arg)
{
    PAS_UNUSED_PARAM(index);
    PAS_UNUSED_PARAM(arg);
    /* Return true if we want to keep going. */
    return *entry_ptr != PAS_BITFIT_MAX_FREE_EMPTY;
}

static PAS_ALWAYS_INLINE pas_found_index
pas_bitfit_directory_find_first_empty(pas_bitfit_directory* directory,
                                      unsigned start_index)
{
    return pas_bitfit_directory_max_free_vector_iterate(
        &directory->max_frees, start_index,
        pas_bitfit_directory_find_first_empty_iterate_callback, NULL);
}

PAS_END_EXTERN_C;

#endif /* PAS_BITFIT_DIRECTORY_INLINES_H */

