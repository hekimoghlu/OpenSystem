/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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
#ifndef PAS_SEGREGATED_PAGE_CONFIG_UTILS_H
#define PAS_SEGREGATED_PAGE_CONFIG_UTILS_H

#include "pas_config.h"
#include "pas_page_base_config_utils.h"
#include "pas_segregated_page.h"
#include "pas_segregated_page_config.h"
#include "pas_segregated_page_inlines.h"
#include "pas_segregated_shared_page_directory.h"
#include "pas_thread_local_cache.h"

PAS_BEGIN_EXTERN_C;

#define PAS_BASIC_SEGREGATED_NUM_ALLOC_BITS(min_align_shift, page_size) \
    ((page_size) >> (min_align_shift))

#define PAS_BASIC_SEGREGATED_PAGE_HEADER_SIZE_EXCLUSIVE(min_align_shift, page_size, granule_size) \
    PAS_SEGREGATED_PAGE_HEADER_SIZE( \
        PAS_BASIC_SEGREGATED_NUM_ALLOC_BITS((min_align_shift), (page_size)), \
        (page_size) / (granule_size))

#define PAS_BASIC_SEGREGATED_PAYLOAD_OFFSET_EXCLUSIVE(min_align_shift, page_size, granule_size) \
    PAS_BASIC_SEGREGATED_PAGE_HEADER_SIZE_EXCLUSIVE((min_align_shift), (page_size), (granule_size))

#define PAS_BASIC_SEGREGATED_PAYLOAD_SIZE_EXCLUSIVE(min_align_shift, page_size, granule_size) \
    ((page_size) - PAS_BASIC_SEGREGATED_PAYLOAD_OFFSET_EXCLUSIVE( \
         (min_align_shift), (page_size), (granule_size)))

#define PAS_BASIC_SEGREGATED_PAGE_HEADER_SIZE_SHARED(min_align_shift, page_size, granule_size) \
    PAS_SEGREGATED_PAGE_HEADER_SIZE( \
        PAS_BASIC_SEGREGATED_NUM_ALLOC_BITS((min_align_shift), (page_size)), \
        (page_size) / (granule_size))

#define PAS_BASIC_SEGREGATED_PAYLOAD_OFFSET_SHARED(min_align_shift, page_size, granule_size) \
    PAS_BASIC_SEGREGATED_PAGE_HEADER_SIZE_SHARED((min_align_shift), (page_size), (granule_size))

#define PAS_BASIC_SEGREGATED_PAYLOAD_SIZE_SHARED(min_align_shift, page_size, granule_size) \
    ((page_size) - PAS_BASIC_SEGREGATED_PAYLOAD_OFFSET_SHARED((min_align_shift), (page_size), (granule_size)))

#define PAS_BASIC_SEGREGATED_PAGE_CONFIG_FORWARD_DECLARATIONS(name) \
    PAS_SEGREGATED_PAGE_CONFIG_SPECIALIZATION_DECLARATIONS(name ## _page_config); \
    PAS_BASIC_PAGE_BASE_CONFIG_FORWARD_DECLARATIONS(name); \
    \
    PAS_API pas_segregated_shared_page_directory* \
    name ## _page_config_select_shared_page_directory( \
        pas_segregated_heap* heap, pas_segregated_size_directory* directory); \

typedef struct {
    pas_page_header_placement_mode header_placement_mode;
    pas_page_header_table* header_table; /* Even if we have multiple tables, this will have one,
                                            since we use this when we know which page config we
                                            are dealing with. */
} pas_basic_segregated_page_config_declarations_arguments;

#define PAS_BASIC_SEGREGATED_PAGE_CONFIG_DECLARATIONS(name, config_value, ...) \
    PAS_BASIC_PAGE_BASE_CONFIG_DECLARATIONS( \
        name, (config_value).base, \
        __VA_ARGS__); \
    \
    struct pas_dummy

PAS_END_EXTERN_C;

#endif /* PAS_SEGREGATED_PAGE_CONFIG_UTILS_H */

