/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 13, 2025.
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
#ifndef PAS_PAGE_BASE_CONFIG_UTILS_INLINES_H
#define PAS_PAGE_BASE_CONFIG_UTILS_INLINES_H

#include "pas_basic_heap_config_enumerator_data.h"
#include "pas_enumerator.h"
#include "pas_heap_lock.h"
#include "pas_page_base_config_utils.h"

PAS_BEGIN_EXTERN_C;

typedef struct {
    pas_page_base_config page_config;
    pas_page_header_table* header_table;
} pas_basic_page_base_config_definitions_arguments;

#define PAS_BASIC_PAGE_BASE_CONFIG_DEFINITIONS(name, ...) \
    pas_page_base* \
    name ## _page_header_for_boundary_remote(pas_enumerator* enumerator, void* boundary) \
    { \
        pas_basic_page_base_config_definitions_arguments arguments = \
            ((pas_basic_page_base_config_definitions_arguments){__VA_ARGS__}); \
        \
        PAS_ASSERT(arguments.page_config.is_enabled); \
        \
        switch (name ## _header_placement_mode) { \
        case pas_page_header_at_head_of_page: { \
            uintptr_t ptr = (uintptr_t)boundary; \
            PAS_PROFILE(PAGE_BASE_FROM_BOUNDARY, ptr); \
            return (pas_page_base*)ptr; \
        } \
        \
        case pas_page_header_in_table: { \
            pas_basic_heap_config_enumerator_data* data; \
            pas_heap_config_kind kind; \
            uintptr_t page_base; \
            \
            kind = arguments.page_config.heap_config_ptr->kind; \
            PAS_ASSERT((unsigned)kind < (unsigned)pas_heap_config_kind_num_kinds); \
            data = (pas_basic_heap_config_enumerator_data*)enumerator->heap_config_datas[kind]; \
            PAS_ASSERT(data); \
            \
            page_base = (uintptr_t)pas_ptr_hash_map_get(&data->page_header_table, boundary).value; \
            PAS_PROFILE(PAGE_BASE_FROM_BOUNDARY, page_base); \
            return (pas_page_base*)page_base; \
        } } \
        \
        PAS_ASSERT(!"Should not be reached"); \
        return NULL; \
    } \
    \
    pas_page_base* name ## _create_page_header( \
        void* boundary, pas_page_kind kind, pas_lock_hold_mode heap_lock_hold_mode) \
    { \
        pas_basic_page_base_config_definitions_arguments arguments = \
            ((pas_basic_page_base_config_definitions_arguments){__VA_ARGS__}); \
        \
        PAS_ASSERT(arguments.page_config.is_enabled); \
        \
        switch (name ## _header_placement_mode) { \
        case pas_page_header_at_head_of_page: { \
            return (pas_page_base*)boundary; \
        } \
        \
        case pas_page_header_in_table: { \
            pas_page_base* result; \
            pas_heap_lock_lock_conditionally(heap_lock_hold_mode); \
            result = pas_page_header_table_add( \
                arguments.header_table, \
                arguments.page_config.page_size, \
                pas_page_base_header_size(arguments.page_config.page_config_ptr, kind), \
                boundary); \
            pas_heap_lock_unlock_conditionally(heap_lock_hold_mode); \
            return result; \
        } } \
        \
        PAS_ASSERT(!"Should not be reached"); \
        return NULL; \
    } \
    \
    void name ## _destroy_page_header( \
        pas_page_base* page, pas_lock_hold_mode heap_lock_hold_mode) \
    { \
        pas_basic_page_base_config_definitions_arguments arguments = \
            ((pas_basic_page_base_config_definitions_arguments){__VA_ARGS__}); \
        \
        PAS_ASSERT(arguments.page_config.is_enabled); \
        \
        switch (name ## _header_placement_mode) { \
        case pas_page_header_at_head_of_page: \
            return; \
        \
        case pas_page_header_in_table: \
            pas_heap_lock_lock_conditionally(heap_lock_hold_mode); \
            pas_page_header_table_remove(arguments.header_table, \
                                         arguments.page_config.page_size, \
                                         page); \
            pas_heap_lock_unlock_conditionally(heap_lock_hold_mode); \
            return; \
        } \
        \
        PAS_ASSERT(!"Should not be reached"); \
        return; \
    } \
    \
    struct pas_dummy

PAS_END_EXTERN_C;

#endif /* PAS_PAGE_BASE_CONFIG_UTILS_INLINES_H */

