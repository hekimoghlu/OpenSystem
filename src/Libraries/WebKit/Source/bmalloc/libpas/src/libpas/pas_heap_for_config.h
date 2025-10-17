/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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
#ifndef PAS_HEAP_FOR_CONFIG_H
#define PAS_HEAP_FOR_CONFIG_H

#include "pas_heap_config.h"
#include "pas_segregated_page_config.h"

PAS_BEGIN_EXTERN_C;

struct pas_allocation_config;
typedef struct pas_allocation_config pas_allocation_config;

PAS_API extern bool pas_heap_for_config_force_bootstrap;

PAS_API void* pas_heap_for_config_allocate(
    const pas_heap_config* config,
    size_t size,
    const char* name);

PAS_API void* pas_heap_for_page_config_kind_allocate(
    pas_segregated_page_config_kind page_config_kind,
    size_t size,
    const char* name);

PAS_API void* pas_heap_for_page_config_allocate(
    const pas_segregated_page_config* page_config,
    size_t size,
    const char* name);

PAS_API void* pas_heap_for_config_allocate_with_alignment(
    const pas_heap_config* config,
    size_t size,
    size_t alignment,
    const char* name);

PAS_API void* pas_heap_for_page_config_allocate_with_alignment(
    const pas_segregated_page_config* page_config,
    size_t size,
    size_t alignment,
    const char* name);

PAS_API void* pas_heap_for_config_allocate_with_manual_alignment(
    const pas_heap_config* config,
    size_t size,
    size_t alignment,
    const char* name);

PAS_API void* pas_heap_for_page_config_kind_allocate_with_manual_alignment(
    pas_segregated_page_config_kind page_config_kind,
    size_t size,
    size_t alignment,
    const char* name);

PAS_API void* pas_heap_for_page_config_allocate_with_manual_alignment(
    const pas_segregated_page_config* page_config,
    size_t size,
    size_t alignment,
    const char* name);

PAS_API void pas_heap_for_config_deallocate(
    const pas_heap_config* config,
    void* ptr,
    size_t size);

PAS_API void pas_heap_for_page_config_kind_deallocate(
    pas_segregated_page_config_kind config_kind,
    void* ptr,
    size_t size);

PAS_API void pas_heap_for_page_config_deallocate(
    const pas_segregated_page_config* config,
    void* ptr,
    size_t size);

PAS_END_EXTERN_C;

#endif /* PAS_HEAP_FOR_CONFIG_H */

