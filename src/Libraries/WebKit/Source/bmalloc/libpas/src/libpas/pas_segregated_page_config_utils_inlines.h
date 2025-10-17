/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 13, 2022.
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
#ifndef PAS_SEGREGATED_PAGE_CONFIG_UTILS_INLINES_H
#define PAS_SEGREGATED_PAGE_CONFIG_UTILS_INLINES_H

#include "pas_config.h"
#include "pas_heap.h"
#include "pas_page_base_config_utils_inlines.h"
#include "pas_segregated_page_config_inlines.h"
#include "pas_segregated_page_config_utils.h"
#include "pas_fast_megapage_cache.h"

PAS_BEGIN_EXTERN_C;

typedef struct {
    pas_segregated_page_config page_config;
    pas_page_header_table* header_table;
} pas_basic_segregated_page_config_definitions_arguments;

#define PAS_BASIC_SEGREGATED_PAGE_CONFIG_DEFINITIONS(name, ...) \
    PAS_SEGREGATED_PAGE_CONFIG_SPECIALIZATION_DEFINITIONS( \
        name ## _page_config, \
        ((pas_basic_segregated_page_config_definitions_arguments){__VA_ARGS__}).page_config); \
    \
    PAS_BASIC_PAGE_BASE_CONFIG_DEFINITIONS( \
        name, \
        ((pas_basic_segregated_page_config_definitions_arguments){__VA_ARGS__}).page_config.base, \
        ((pas_basic_segregated_page_config_definitions_arguments){__VA_ARGS__}).header_table); \
    \
    pas_segregated_shared_page_directory* \
    name ## _page_config_select_shared_page_directory( \
        pas_segregated_heap* heap, pas_segregated_size_directory* size_directory) \
    { \
        PAS_UNUSED_PARAM(size_directory); \
        \
        pas_basic_segregated_page_config_definitions_arguments arguments = \
            (pas_basic_segregated_page_config_definitions_arguments){__VA_ARGS__}; \
        \
        pas_shared_page_directory_by_size* directory_by_size; \
        pas_segregated_shared_page_directory* directory; \
        \
        PAS_ASSERT(arguments.page_config.base.is_enabled); \
        \
        directory_by_size = pas_basic_heap_page_caches_get_shared_page_directories( \
            ((pas_basic_heap_runtime_config*)heap->runtime_config)->page_caches, \
            arguments.page_config.variant); \
        \
        directory = pas_shared_page_directory_by_size_get( \
            directory_by_size, size_directory->object_size, \
            (const pas_segregated_page_config*)arguments.page_config.base.page_config_ptr); \
        \
        return directory; \
    } \
    \
    struct pas_dummy

PAS_END_EXTERN_C;

#endif /* PAS_SEGREGATED_PAGE_CONFIG_UTILS_INLINES_H */

