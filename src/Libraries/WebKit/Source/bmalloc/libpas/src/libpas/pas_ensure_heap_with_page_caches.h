/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 22, 2025.
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
#ifndef PAS_ENSURE_HEAP_WITH_PAGE_CACHES_H
#define PAS_ENSURE_HEAP_WITH_PAGE_CACHES_H

#include "pas_heap_ref.h"

PAS_BEGIN_EXTERN_C;

struct pas_basic_heap_page_caches;
typedef struct pas_basic_heap_page_caches pas_basic_heap_page_caches;

/* To call this function, the heap_ref must still not be initialized. Also, the heap must be
   one of the "basic" ones - created with pas_heap_config_utils or something that broadly uses
   the same defaults. In particular, it must be the kind of heap that expects the runtime_config
   to be a pas_basic_heap_runtime_config. This will copy the runtime_config you pass and combine
   it with the basic_heap_page_caches to create a new pas_basic_heap_runtime_config. */
PAS_API pas_heap* pas_ensure_heap_with_page_caches(
    pas_heap_ref* heap_ref,
    pas_heap_ref_kind heap_ref_kind,
    const pas_heap_config* config,
    pas_heap_runtime_config* template_runtime_config,
    pas_basic_heap_page_caches* page_caches);

PAS_END_EXTERN_C;

#endif /* PAS_ENSURE_HEAP_WITH_PAGE_CACHES_H */

