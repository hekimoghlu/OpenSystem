/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 4, 2025.
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
#ifndef PAS_BITFIT_HEAP_H
#define PAS_BITFIT_HEAP_H

#include "pas_bitfit_directory.h"
#include "pas_bitfit_page_config_variant.h"

PAS_BEGIN_EXTERN_C;

struct pas_bitfit_heap;
struct pas_heap_config;
struct pas_heap_runtime_config;
struct pas_segregated_size_directory;
struct pas_segregated_heap;
typedef struct pas_bitfit_heap pas_bitfit_heap;
typedef struct pas_heap_config pas_heap_config;
typedef struct pas_heap_runtime_config pas_heap_runtime_config;
typedef struct pas_segregated_size_directory pas_segregated_size_directory;
typedef struct pas_segregated_heap pas_segregated_heap;

struct PAS_ALIGNED(sizeof(pas_versioned_field)) pas_bitfit_heap {
    pas_bitfit_directory directories[PAS_NUM_BITFIT_PAGE_CONFIG_VARIANTS];
};

PAS_API pas_bitfit_heap* pas_bitfit_heap_create(pas_segregated_heap* heap,
                                                const pas_heap_config* heap_config);

static inline pas_bitfit_directory* pas_bitfit_heap_get_directory(
    pas_bitfit_heap* heap,
    pas_bitfit_page_config_variant variant)
{
    PAS_ASSERT((unsigned)variant <= PAS_NUM_BITFIT_PAGE_CONFIG_VARIANTS);
    return heap->directories + (unsigned)variant;
}

typedef struct {
    unsigned object_size;
    pas_bitfit_page_config_variant variant;
} pas_bitfit_variant_selection;

PAS_API pas_bitfit_variant_selection
pas_bitfit_heap_select_variant(size_t object_size,
                               const pas_heap_config* config,
                               pas_heap_runtime_config* runtime_config);

PAS_API void pas_bitfit_heap_construct_and_insert_size_class(pas_bitfit_heap* heap,
                                                             pas_bitfit_size_class* size_class,
                                                             unsigned object_size,
                                                             const pas_heap_config* config,
                                                             pas_heap_runtime_config* runtime_config);

PAS_API pas_heap_summary pas_bitfit_heap_compute_summary(pas_bitfit_heap* heap);

typedef bool (*pas_bitfit_heap_for_each_live_object_callback)(
    pas_bitfit_heap* heap,
    pas_bitfit_view* view,
    uintptr_t begin,
    size_t size,
    void* arg);

PAS_API bool pas_bitfit_heap_for_each_live_object(
    pas_bitfit_heap* heap,
    pas_bitfit_heap_for_each_live_object_callback callback,
    void* arg);

PAS_END_EXTERN_C;

#endif /* PAS_BITFIT_HEAP_H */

