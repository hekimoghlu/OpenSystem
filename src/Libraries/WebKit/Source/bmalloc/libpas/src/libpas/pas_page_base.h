/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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
#ifndef PAS_PAGE_BASE_H
#define PAS_PAGE_BASE_H

#include "pas_free_range_kind.h"
#include "pas_page_base_config.h"
#include "pas_page_config_kind.h"
#include "pas_page_kind.h"
#include "pas_range.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_bitfit_page;
struct pas_heap_summary;
struct pas_page_base;
struct pas_segregated_page;
typedef struct pas_bitfit_page pas_bitfit_page;
typedef struct pas_heap_summary pas_heap_summary;
typedef struct pas_page_base pas_page_base;
typedef struct pas_segregated_page pas_segregated_page;

struct pas_page_base {
    uint8_t page_kind; /* This cannot be the enum because then the alignment would be 4.
                          Note that we don't *really* need this, but:
                       
                          - It's convenient. It's nice to be able to use it at least for assertions
                            even if for nothing else.
                       
                          - It costs nothing. Both segregated_page and bitfit_page have a byte to
                            spare at the beginning.
                       
                          Because this only exists for convenience and because it currently costs
                          nothing, we should get rid of this field and remove page_base as a field
                          from segregated_page and bitfit_page if we ever did need that space for
                          basically any reason. */
};

PAS_API size_t pas_page_base_header_size(const pas_page_base_config* config,
                                         pas_page_kind page_kind);

static inline void pas_page_base_construct(pas_page_base* page_base,
                                           pas_page_kind page_kind)
{
    page_base->page_kind = (uint8_t)page_kind;
}

static inline pas_page_kind pas_page_base_get_kind(pas_page_base* page_base)
{
    return (pas_page_kind)page_base->page_kind;
}

static inline pas_page_config_kind pas_page_base_get_config_kind(pas_page_base* page_base)
{
    return pas_page_kind_get_config_kind((pas_page_kind)page_base->page_kind);
}

static inline bool pas_page_base_is_segregated(pas_page_base* page_base)
{
    return pas_page_base_get_config_kind(page_base) == pas_page_config_kind_segregated;
}

static inline pas_segregated_page* pas_page_base_get_segregated(pas_page_base* page_base)
{
    PAS_TESTING_ASSERT(!page_base || pas_page_base_is_segregated(page_base));
    return (pas_segregated_page*)page_base;
}

static inline bool pas_page_base_is_bitfit(pas_page_base* page_base)
{
    return pas_page_base_get_config_kind(page_base) == pas_page_config_kind_bitfit;
}

static inline pas_bitfit_page* pas_page_base_get_bitfit(pas_page_base* page_base)
{
    PAS_TESTING_ASSERT(!page_base || pas_page_base_is_bitfit(page_base));
    return (pas_bitfit_page*)page_base;
}

static PAS_ALWAYS_INLINE uintptr_t
pas_page_base_index_of_object_at_offset_from_page_boundary(
    uintptr_t offset_in_page,
    pas_page_base_config page_config)
{
    return offset_in_page >> page_config.min_align_shift;
}

static PAS_ALWAYS_INLINE uintptr_t
pas_page_base_object_offset_from_page_boundary_at_index(
    uintptr_t index,
    pas_page_base_config page_config)
{
    return index << page_config.min_align_shift;
}

static PAS_ALWAYS_INLINE void* pas_page_base_boundary(
    pas_page_base* page,
    pas_page_base_config page_config)
{
    PAS_TESTING_ASSERT(page);
    return page_config.boundary_for_page_header(page);
}

static PAS_ALWAYS_INLINE void* pas_page_base_boundary_or_null(
    pas_page_base* page,
    pas_page_base_config page_config)
{
    if (!page)
        return NULL;
    return pas_page_base_boundary(page, page_config);
}

static PAS_ALWAYS_INLINE pas_page_base*
pas_page_base_for_boundary(void* boundary,
                           pas_page_base_config page_config)
{
    PAS_TESTING_ASSERT(boundary);
    return page_config.page_header_for_boundary(boundary);
}

static PAS_ALWAYS_INLINE pas_page_base*
pas_page_base_for_boundary_or_null(void* boundary,
                                   pas_page_base_config page_config)
{
    if (!boundary)
        return NULL;
    return pas_page_base_for_boundary(boundary, page_config);
}

static PAS_ALWAYS_INLINE void* pas_page_base_boundary_for_address_and_page_config(
    uintptr_t begin,
    pas_page_base_config page_config)
{
    return (void*)pas_round_down_to_power_of_2(begin, page_config.page_size);
}

static PAS_ALWAYS_INLINE pas_page_base*
pas_page_base_for_address_and_page_config(uintptr_t begin,
                                          pas_page_base_config page_config)
{
    return pas_page_base_for_boundary(
        pas_page_base_boundary_for_address_and_page_config(begin, page_config),
        page_config);
}

PAS_API const pas_page_base_config* pas_page_base_get_config(pas_page_base* page);

PAS_API pas_page_granule_use_count*
pas_page_base_get_granule_use_counts(pas_page_base* page);

PAS_API void pas_page_base_compute_committed_when_owned(pas_page_base* page,
                                                        pas_heap_summary* summary);

PAS_API bool pas_page_base_is_empty(pas_page_base* page);

PAS_API void pas_page_base_add_free_range(pas_page_base* page,
                                          pas_heap_summary* result,
                                          pas_range range,
                                          pas_free_range_kind kind);

PAS_END_EXTERN_C;

#endif /* PAS_PAGE_BASE_H */

