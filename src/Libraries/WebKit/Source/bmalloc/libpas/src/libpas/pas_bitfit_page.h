/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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
#ifndef PAS_BITFIT_PAGE_H
#define PAS_BITFIT_PAGE_H

#include "pas_bitfit_page_config.h"
#include "pas_compact_atomic_bitfit_view_ptr.h"
#include "pas_lock.h"
#include "pas_page_base.h"

PAS_BEGIN_EXTERN_C;

struct pas_bitfit_page;
typedef struct pas_bitfit_page pas_bitfit_page;

struct pas_bitfit_page {
    pas_page_base base;

    /* Tells us if we've told the directory that something was freed. If true, we ignore freeing.
       If false, we tell the directory that the page may have any amount of free bytes. The
       allocator will lazily figure out exactly how many free bytes later. */
    bool did_note_max_free;

    /* This is a source of truth - for example, we may have a view that says that the page is
       empty but this says it's not. */
    uint16_t num_live_bits;

    pas_compact_atomic_bitfit_view_ptr owner;

    uint64_t use_epoch;

    uint64_t bits[1]; /* This gets accessed in a crazy way. */
};

#define PAS_BITFIT_PAGE_BASE_HEADER_SIZE PAS_OFFSETOF(pas_bitfit_page, bits)

#define PAS_BITFIT_PAGE_HEADER_SIZE(page_size, granule_size, min_align_shift) \
    (PAS_BITFIT_PAGE_BASE_HEADER_SIZE + \
     PAS_ROUND_UP_TO_POWER_OF_2( \
         PAS_BITFIT_PAGE_CONFIG_NUM_ALLOC_BIT_BYTES(page_size, min_align_shift) + \
         PAS_PAGE_BASE_CONFIG_NUM_GRANULE_BYTES(page_size / granule_size), \
         sizeof(uint64_t)))

static PAS_ALWAYS_INLINE size_t pas_bitfit_page_header_size(pas_bitfit_page_config page_config)
{
    return PAS_BITFIT_PAGE_HEADER_SIZE(page_config.base.page_size,
                                       page_config.base.granule_size,
                                       page_config.base.min_align_shift);
}

static inline unsigned* pas_bitfit_page_free_bits(pas_bitfit_page* page)
{
    return (unsigned*)page->bits;
}

static PAS_ALWAYS_INLINE unsigned* pas_bitfit_page_object_end_bits(pas_bitfit_page* page,
                                                                   pas_bitfit_page_config page_config)
{
    return (unsigned*)(
        (char*)page->bits + pas_bitfit_page_config_byte_offset_for_object_bits(page_config));
}

static PAS_ALWAYS_INLINE pas_page_granule_use_count*
pas_bitfit_page_get_granule_use_counts(pas_bitfit_page* page,
                                       pas_bitfit_page_config page_config)
{
    PAS_ASSERT(page_config.base.page_size > page_config.base.granule_size);
    return (pas_page_granule_use_count*)(
        (char*)page->bits + pas_bitfit_page_config_num_alloc_bit_bytes(page_config));
}

PAS_API void pas_bitfit_page_construct(pas_bitfit_page* page,
                                       pas_bitfit_view* view,
                                       const pas_bitfit_page_config* config);

static PAS_ALWAYS_INLINE uintptr_t
pas_bitfit_page_offset_to_first_object(pas_bitfit_page_config page_config)
{
    return pas_round_up_to_power_of_2(page_config.page_object_payload_offset,
                                      pas_page_base_config_min_align(page_config.base));
}

static PAS_ALWAYS_INLINE uintptr_t
pas_bitfit_page_offset_to_end_of_last_object(pas_bitfit_page_config page_config)
{
    return pas_round_down_to_power_of_2(
        pas_bitfit_page_config_object_payload_end_offset_from_boundary(page_config),
        pas_page_base_config_min_align(page_config.base));
}

static PAS_ALWAYS_INLINE size_t
pas_bitfit_page_payload_size(pas_bitfit_page_config page_config)
{
    uintptr_t first_object;
    uintptr_t end_of_last_object;

    first_object = pas_bitfit_page_offset_to_first_object(page_config);
    end_of_last_object = pas_bitfit_page_offset_to_end_of_last_object(page_config);
    
    PAS_ASSERT(first_object < end_of_last_object);

    return end_of_last_object - first_object;
}

static PAS_ALWAYS_INLINE pas_bitfit_page*
pas_bitfit_page_for_boundary(void* boundary,
                             pas_bitfit_page_config page_config)
{
    return pas_page_base_get_bitfit(pas_page_base_for_boundary(boundary, page_config.base));
}

static PAS_ALWAYS_INLINE pas_bitfit_page*
pas_bitfit_page_for_boundary_or_null(void* boundary,
                                     pas_bitfit_page_config page_config)
{
    return pas_page_base_get_bitfit(pas_page_base_for_boundary_or_null(boundary, page_config.base));
}

static PAS_ALWAYS_INLINE pas_bitfit_page*
pas_bitfit_page_for_boundary_unchecked(void* boundary,
                                       pas_bitfit_page_config page_config)
{
    return (pas_bitfit_page*)pas_page_base_for_boundary(boundary, page_config.base);
}

static PAS_ALWAYS_INLINE void* pas_bitfit_page_boundary(pas_bitfit_page* page,
                                                        pas_bitfit_page_config page_config)
{
    return pas_page_base_boundary(&page->base, page_config.base);
}

static PAS_ALWAYS_INLINE void* pas_bitfit_page_boundary_or_null(pas_bitfit_page* page,
                                                                pas_bitfit_page_config page_config)
{
    return pas_page_base_boundary_or_null(&page->base, page_config.base);
}

static PAS_ALWAYS_INLINE pas_bitfit_page*
pas_bitfit_page_for_address_and_page_config(uintptr_t begin,
                                            pas_bitfit_page_config page_config)
{
    return pas_page_base_get_bitfit(
        pas_page_base_for_address_and_page_config(begin, page_config.base));
}

PAS_API const pas_bitfit_page_config* pas_bitfit_page_get_config(pas_bitfit_page* page);

PAS_API void pas_bitfit_page_log_bits(
    pas_bitfit_page* page, uintptr_t mark_begin_offset, uintptr_t mark_end_offset);

PAS_API PAS_NO_RETURN void pas_bitfit_page_deallocation_did_fail(
    pas_bitfit_page* page,
    pas_bitfit_page_config_kind config_kind,
    uintptr_t begin,
    uintptr_t offset,
    const char* reason);

typedef bool (*pas_bitfit_page_for_each_live_object_callback)(
    uintptr_t begin,
    size_t size,
    void* arg);

PAS_API bool pas_bitfit_page_for_each_live_object(
    pas_bitfit_page* page,
    pas_bitfit_page_for_each_live_object_callback callback,
    void* arg);

PAS_API void pas_bitfit_page_verify(pas_bitfit_page* page);

static inline void pas_bitfit_page_testing_verify(pas_bitfit_page* page)
{
    if (PAS_ENABLE_TESTING)
        pas_bitfit_page_verify(page);
}

PAS_END_EXTERN_C;

#endif /* PAS_BITFIT_PAGE_H */

