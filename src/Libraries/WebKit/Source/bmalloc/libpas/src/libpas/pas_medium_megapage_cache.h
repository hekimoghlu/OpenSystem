/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 4, 2023.
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
#ifndef PAS_MEDIUM_MEGAPAGE_CACHE_H
#define PAS_MEDIUM_MEGAPAGE_CACHE_H

#include "pas_heap_ref.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_megapage_cache;
struct pas_page_base_config;
struct pas_physical_memory_transaction;
typedef struct pas_megapage_cache pas_megapage_cache;
typedef struct pas_page_base_config pas_page_base_config;
typedef struct pas_physical_memory_transaction pas_physical_memory_transaction;

/* Allocates a small page aligned the way that the type-safe heap demands and returns the
   base on the allocation. This is guaranteed to returned zeroed memory. */
PAS_API void* pas_medium_megapage_cache_try_allocate(
    pas_megapage_cache* cache,
    const pas_page_base_config* config,
    bool should_zero,
    pas_heap* heap,
    pas_physical_memory_transaction* transaction);

PAS_END_EXTERN_C;

#endif /* PAS_MEDIUM_MEGAPAGE_CACHE_H */

