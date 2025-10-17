/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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

#include "pas_page_base.h"

#include "pas_bitfit_page.h"
#include "pas_segregated_page.h"

size_t pas_page_base_header_size(const pas_page_base_config* config,
                                 pas_page_kind page_kind)
{
    switch (config->page_config_kind) {
    case pas_page_config_kind_segregated:
        PAS_ASSERT(pas_page_kind_get_config_kind(page_kind) == pas_page_config_kind_segregated);
        return pas_segregated_page_header_size(
            *pas_page_base_config_get_segregated(config),
            pas_page_kind_get_segregated_role(page_kind));
    case pas_page_config_kind_bitfit:
        PAS_ASSERT(pas_page_kind_get_config_kind(page_kind) == pas_page_config_kind_bitfit);
        return pas_bitfit_page_header_size(*pas_page_base_config_get_bitfit(config));
    }
    PAS_ASSERT(!"Should not be reached");
    return 0;
}

const pas_page_base_config* pas_page_base_get_config(pas_page_base* page)
{
    switch (pas_page_base_get_config_kind(page)) {
    case pas_page_config_kind_segregated:
        return &pas_segregated_page_get_config(pas_page_base_get_segregated(page))->base;
    case pas_page_config_kind_bitfit:
        return &pas_bitfit_page_get_config(pas_page_base_get_bitfit(page))->base;
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

pas_page_granule_use_count*
pas_page_base_get_granule_use_counts(pas_page_base* page)
{
    switch (pas_page_base_get_config_kind(page)) {
    case pas_page_config_kind_segregated: {
        pas_segregated_page* segregated_page;
        segregated_page = pas_page_base_get_segregated(page);
        return pas_segregated_page_get_granule_use_counts(
            segregated_page, *pas_segregated_page_get_config(segregated_page));
    }
    case pas_page_config_kind_bitfit: {
        pas_bitfit_page* bitfit_page;
        bitfit_page = pas_page_base_get_bitfit(page);
        return pas_bitfit_page_get_granule_use_counts(
            bitfit_page, *pas_bitfit_page_get_config(bitfit_page));
    } }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

void pas_page_base_compute_committed_when_owned(pas_page_base* page,
                                                pas_heap_summary* summary)
{
    pas_page_granule_use_count* use_counts;
    uintptr_t num_granules;
    uintptr_t granule_index;
    const pas_page_base_config* config_ptr;
    pas_page_base_config config;

    config_ptr = pas_page_base_get_config(page);
    config = *config_ptr;
    
    if (config.page_size == config.granule_size) {
        summary->committed += config.page_size;
        return;
    }
    
    use_counts = pas_page_base_get_granule_use_counts(page);
    num_granules = config.page_size / config.granule_size;
    
    for (granule_index = num_granules; granule_index--;) {
        if (use_counts[granule_index] == PAS_PAGE_GRANULE_DECOMMITTED)
            summary->decommitted += config.granule_size;
        else
            summary->committed += config.granule_size;
    }
}

bool pas_page_base_is_empty(pas_page_base* page)
{
    switch (pas_page_base_get_config_kind(page)) {
    case pas_page_config_kind_segregated:
        return !pas_page_base_get_segregated(page)->emptiness.num_non_empty_words;
    case pas_page_config_kind_bitfit:
        return !pas_page_base_get_bitfit(page)->num_live_bits;
    }
    PAS_ASSERT(!"Should not be reached");
    return false;
}

void pas_page_base_add_free_range(pas_page_base* page,
                                  pas_heap_summary* result,
                                  pas_range range,
                                  pas_free_range_kind kind)
{
    const pas_page_base_config* page_config_ptr;
    pas_page_base_config page_config;
    size_t* ineligible_for_decommit;
    size_t* eligible_for_decommit;
    size_t* decommitted;
    size_t dummy;
    bool empty;
    pas_page_granule_use_count* use_counts;
    uintptr_t first_granule_index;
    uintptr_t last_granule_index;
    uintptr_t granule_index;

    if (pas_range_is_empty(range))
        return;

    PAS_ASSERT(range.end > range.begin);

    page_config_ptr = pas_page_base_get_config(page);
    page_config = *page_config_ptr;

    PAS_ASSERT(range.end <= page_config.page_size);

    empty = pas_page_base_is_empty(page);

    dummy = 0; /* Tell the compiler to chill out and relax. */

    switch (kind) {
    case pas_free_object_range:
        result->free += pas_range_size(range);
        
        ineligible_for_decommit = &result->free_ineligible_for_decommit;
        eligible_for_decommit = &result->free_eligible_for_decommit;
        decommitted = &result->free_decommitted;
        break;
    case pas_free_meta_range:
        result->meta += pas_range_size(range);
        
        ineligible_for_decommit = &result->meta_ineligible_for_decommit;
        eligible_for_decommit = &result->meta_eligible_for_decommit;
        decommitted = &dummy;
        break;
    }
    
    if (page_config.page_size == page_config.granule_size) {
        if (empty)
            (*eligible_for_decommit) += pas_range_size(range);
        else
            (*ineligible_for_decommit) += pas_range_size(range);
        return;
    }
    
    use_counts = pas_page_base_get_granule_use_counts(page);

    first_granule_index = range.begin / page_config.granule_size;
    last_granule_index = (range.end - 1) / page_config.granule_size;

    for (granule_index = first_granule_index;
         granule_index <= last_granule_index;
         ++granule_index) {
        pas_range granule_range;
        size_t overlap_size;
        
        granule_range = pas_range_create(granule_index * page_config.granule_size,
                                         (granule_index + 1) * page_config.granule_size);

        PAS_ASSERT(pas_range_overlaps(range, granule_range));

        overlap_size = pas_range_size(pas_range_create_intersection(range,
                                                                    granule_range));

        switch (use_counts[granule_index]) {
        case 0:
            *eligible_for_decommit += overlap_size;
            break;
        case PAS_PAGE_GRANULE_DECOMMITTED:
            *decommitted += overlap_size;
            break;
        default:
            *ineligible_for_decommit += overlap_size;
            break;
        }
    }
}

#endif /* LIBPAS_ENABLED */
