/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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
#ifndef PAS_PAGE_BASE_CONFIG_UTILS_H
#define PAS_PAGE_BASE_CONFIG_UTILS_H

#include "pas_config.h"
#include "pas_internal_config.h"
#include "pas_page_base_config.h"
#include "pas_page_header_placement_mode.h"
#include "pas_page_header_table.h"

PAS_BEGIN_EXTERN_C;

#define PAS_BASIC_PAGE_BASE_CONFIG_FORWARD_DECLARATIONS(name) \
    static PAS_ALWAYS_INLINE pas_page_base* \
    name ## _page_header_for_boundary(void* boundary); \
    static PAS_ALWAYS_INLINE void* \
    name ## _boundary_for_page_header(pas_page_base* page); \
    PAS_API pas_page_base* \
    name ## _page_header_for_boundary_remote(pas_enumerator* enumerator, void* boundary); \
    \
    PAS_API pas_page_base* name ## _create_page_header( \
        void* boundary, pas_page_kind kind, pas_lock_hold_mode heap_lock_hold_mode); \
    PAS_API void name ## _destroy_page_header( \
        pas_page_base* page, pas_lock_hold_mode heap_lock_hold_mode)

typedef struct {
    pas_page_header_placement_mode header_placement_mode;
    pas_page_header_table* header_table; /* Even if we have multiple tables, this will have one,
                                            since we use this when we know which page config we
                                            are dealing with. */
} pas_basic_page_base_config_declarations_arguments;

#define PAS_BASIC_PAGE_BASE_CONFIG_DECLARATIONS(name, config_value, header_placement_mode_value, header_table_value) \
    static const pas_page_header_placement_mode name ## _header_placement_mode = (header_placement_mode_value); \
    static PAS_ALWAYS_INLINE pas_page_base* \
    name ## _page_header_for_boundary(void* boundary) \
    { \
        pas_basic_page_base_config_declarations_arguments arguments = { .header_placement_mode = (header_placement_mode_value), .header_table = (header_table_value) }; \
        pas_page_base_config config; \
        \
        config = (config_value); \
        PAS_ASSERT(config.is_enabled); \
        \
        switch (arguments.header_placement_mode) { \
        case pas_page_header_at_head_of_page: { \
            uintptr_t ptr = (uintptr_t)boundary; \
            PAS_PROFILE(PAGE_BASE_FROM_BOUNDARY, ptr); \
            return (pas_page_base*)ptr; \
        } \
        \
        case pas_page_header_in_table: { \
            uintptr_t page_base; \
            \
            page_base = (uintptr_t)pas_page_header_table_get_for_boundary( \
                arguments.header_table, config.page_size, boundary); \
            PAS_TESTING_ASSERT(page_base); \
            PAS_PROFILE(PAGE_BASE_FROM_TABLE, page_base); \
            return (pas_page_base*)page_base; \
        } } \
        \
        PAS_ASSERT(!"Should not be reached"); \
        return NULL; \
    } \
    \
    static PAS_ALWAYS_INLINE void* \
    name ## _boundary_for_page_header(pas_page_base* page) \
    { \
        pas_basic_page_base_config_declarations_arguments arguments = { .header_placement_mode = (header_placement_mode_value), .header_table = (header_table_value) }; \
        pas_page_base_config config; \
        \
        config = (config_value); \
        PAS_ASSERT(config.is_enabled); \
        \
        switch (name ## _header_placement_mode) { \
        case pas_page_header_at_head_of_page: { \
            return page; \
        } \
        \
        case pas_page_header_in_table: { \
            void* boundary; \
            \
            boundary = pas_page_header_table_get_boundary( \
                arguments.header_table, config.page_size, page); \
            PAS_TESTING_ASSERT(boundary); \
            return boundary; \
        } } \
        \
        PAS_ASSERT(!"Should not be reached"); \
        return NULL; \
    } \
    \
    struct pas_dummy

    PAS_END_EXTERN_C;

#endif /* PAS_PAGE_BASE_CONFIG_UTILS_H */

