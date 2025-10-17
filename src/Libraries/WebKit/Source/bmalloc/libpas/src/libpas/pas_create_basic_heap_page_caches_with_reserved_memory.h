/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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
#ifndef PAS_CREATE_BASIC_HEAP_PAGE_CACHES_WITH_RESERVED_MEMORY_H
#define PAS_CREATE_BASIC_HEAP_PAGE_CACHES_WITH_RESERVED_MEMORY_H

#include "pas_heap.h"

PAS_BEGIN_EXTERN_C;

struct pas_basic_heap_page_caches;
struct pas_basic_heap_runtime_config;
typedef struct pas_basic_heap_page_caches pas_basic_heap_page_caches;
typedef struct pas_basic_heap_runtime_config pas_basic_heap_runtime_config;

/* Warning: This creates caches that allow type confusion. Only use this for primitive heaps! */
PAS_API pas_basic_heap_page_caches* pas_create_basic_heap_page_caches_with_reserved_memory(
    pas_basic_heap_runtime_config* template_runtime_config,
    uintptr_t begin,
    uintptr_t end);

PAS_END_EXTERN_C;

#endif /* PAS_CREATE_BASIC_HEAP_PAGE_CACHES_WITH_RESERVED_MEMORY_H */

