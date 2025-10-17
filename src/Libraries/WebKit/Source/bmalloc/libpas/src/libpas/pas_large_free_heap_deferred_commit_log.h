/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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
#ifndef PAS_LARGE_FREE_HEAP_DEFERRED_COMMIT_LOG_H
#define PAS_LARGE_FREE_HEAP_DEFERRED_COMMIT_LOG_H

#include "pas_large_virtual_range.h"
#include "pas_large_virtual_range_min_heap.h"

PAS_BEGIN_EXTERN_C;

struct pas_large_free_heap_deferred_commit_log;
struct pas_physical_memory_transaction;
typedef struct pas_large_free_heap_deferred_commit_log pas_large_free_heap_deferred_commit_log;
typedef struct pas_physical_memory_transaction pas_physical_memory_transaction;

struct pas_large_free_heap_deferred_commit_log {
    pas_large_virtual_range_min_heap impl;
    size_t total; /* This is accurate so long as the ranges are non-overlapping. */
};

PAS_API void pas_large_free_heap_deferred_commit_log_construct(
    pas_large_free_heap_deferred_commit_log* log);

PAS_API void pas_large_free_heap_deferred_commit_log_destruct(
    pas_large_free_heap_deferred_commit_log* log);

PAS_API bool pas_large_free_heap_deferred_commit_log_add(
    pas_large_free_heap_deferred_commit_log* log,
    pas_large_virtual_range range,
    pas_physical_memory_transaction* transaction);

PAS_API void pas_large_free_heap_deferred_commit_log_commit_all(
    pas_large_free_heap_deferred_commit_log* log,
    pas_physical_memory_transaction* transaction);

/* Useful for writing tests. */
PAS_API void pas_large_free_heap_deferred_commit_log_pretend_to_commit_all(
    pas_large_free_heap_deferred_commit_log* log,
    pas_physical_memory_transaction* transaction);

PAS_END_EXTERN_C;

#endif /* PAS_LARGE_FREE_HEAP_DEFERRED_COMMIT_LOG_H */

