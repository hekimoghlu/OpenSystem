/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
#ifndef PAS_BITFIT_PAGE_CONFIG_INLINES_H
#define PAS_BITFIT_PAGE_CONFIG_INLINES_H

#include "pas_bitfit_allocator_inlines.h"
#include "pas_bitfit_page_config.h"
#include "pas_bitfit_page_inlines.h"

PAS_BEGIN_EXTERN_C;

#define PAS_BITFIT_PAGE_CONFIG_SPECIALIZATION_DEFINITIONS(lower_case_page_config_name, page_config_value) \
    pas_fast_path_allocation_result \
    lower_case_page_config_name ## _specialized_allocator_try_allocate( \
        pas_bitfit_allocator* allocator, \
        pas_local_allocator* local_allocator, \
        size_t size, \
        size_t alignment, \
        pas_allocation_mode allocation_mode) \
    { \
        return pas_bitfit_allocator_try_allocate( \
            allocator, local_allocator, size, alignment, allocation_mode, (page_config_value)); \
    } \
    \
    void lower_case_page_config_name ## _specialized_page_deallocate_with_page( \
        pas_bitfit_page* page, uintptr_t begin) \
    { \
        pas_bitfit_page_deallocate_with_page(page, begin, (page_config_value)); \
    } \
    \
    size_t lower_case_page_config_name ## _specialized_page_get_allocation_size_with_page( \
        pas_bitfit_page* page, uintptr_t begin) \
    { \
        return pas_bitfit_page_get_allocation_size_with_page(page, begin, (page_config_value)); \
    } \
    \
    void lower_case_page_config_name ## _specialized_page_shrink_with_page( \
        pas_bitfit_page* page, uintptr_t begin, size_t new_size) \
    { \
        return pas_bitfit_page_shrink_with_page(page, begin, new_size, (page_config_value)); \
    } \
    \
    struct pas_dummy

PAS_END_EXTERN_C;

#endif /* PAS_BITFIT_PAGE_CONFIG_INLINES_H */

