/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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
#ifndef PAS_COMMIT_SPAN_H
#define PAS_COMMIT_SPAN_H

#include "pas_lock.h"
#include "pas_mmap_capability.h"

PAS_BEGIN_EXTERN_C;

struct pas_commit_span;
struct pas_deferred_decommit_log;
struct pas_page_base;
struct pas_page_base_config;
typedef struct pas_commit_span pas_commit_span;
typedef struct pas_deferred_decommit_log pas_deferred_decommit_log;
typedef struct pas_page_base pas_page_base;
typedef struct pas_page_base_config pas_page_base_config;

struct pas_commit_span {
    uintptr_t index_of_start_of_span;
    bool did_add_first;
    size_t total_bytes;
    pas_mmap_capability mmap_capability;
};

PAS_API void pas_commit_span_construct(pas_commit_span* span, pas_mmap_capability mmap_capability);
PAS_API void pas_commit_span_add_to_change(pas_commit_span* span, uintptr_t granule_index);
PAS_API void pas_commit_span_add_unchanged(pas_commit_span* span,
                                           pas_page_base* page,
                                           uintptr_t granule_index,
                                           const pas_page_base_config* config,
                                           void (*commit_or_decommit)(
                                               void* base, size_t size, void* arg),
                                           void* arg);
PAS_API void pas_commit_span_add_unchanged_and_commit(pas_commit_span* span,
                                                      pas_page_base* page,
                                                      uintptr_t granule_index,
                                                      const pas_page_base_config* config);
PAS_API void pas_commit_span_add_unchanged_and_decommit(pas_commit_span* span,
                                                        pas_page_base* page,
                                                        uintptr_t granule_index,
                                                        pas_deferred_decommit_log* log,
                                                        pas_lock* commit_lock,
                                                        const pas_page_base_config* config,
                                                        pas_lock_hold_mode heap_lock_hold_mode);

PAS_END_EXTERN_C;

#endif /* PAS_COMMIT_SPAN_H */

