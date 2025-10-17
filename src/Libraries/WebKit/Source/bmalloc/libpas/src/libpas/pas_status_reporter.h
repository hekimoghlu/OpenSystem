/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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
#ifndef PAS_STATUS_REPORTER_H
#define PAS_STATUS_REPORTER_H

#include "pas_heap.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_bitfit_directory;
struct pas_large_heap;
struct pas_segregated_heap;
struct pas_segregated_shared_page_directory;
struct pas_segregated_size_directory;
struct pas_stream;
typedef struct pas_bitfit_directory pas_bitfit_directory;
typedef struct pas_large_heap pas_large_heap;
typedef struct pas_segregated_heap pas_segregated_heap;
typedef struct pas_segregated_shared_page_directory pas_segregated_shared_page_directory;
typedef struct pas_segregated_size_directory pas_segregated_size_directory;
typedef struct pas_stream pas_stream;

PAS_API extern unsigned pas_status_reporter_enabled;
PAS_API extern unsigned pas_status_reporter_period_in_microseconds;

PAS_API void pas_status_reporter_dump_bitfit_directory(
    pas_stream* stream, pas_bitfit_directory* directory);
PAS_API void pas_status_reporter_dump_segregated_size_directory(
    pas_stream* stream, pas_segregated_size_directory* directory);
PAS_API void pas_status_reporter_dump_segregated_shared_page_directory(
    pas_stream* stream, pas_segregated_shared_page_directory* directory);
PAS_API void pas_status_reporter_dump_large_heap(pas_stream* stream, pas_large_heap* heap);
PAS_API void pas_status_reporter_dump_large_map(pas_stream* stream);
PAS_API void pas_status_reporter_dump_heap_table(pas_stream* stream);
PAS_API void pas_status_reporter_dump_immortal_heap(pas_stream* stream);
PAS_API void pas_status_reporter_dump_compact_large_utility_free_heap(pas_stream* stream);
PAS_API void pas_status_reporter_dump_large_utility_free_heap(pas_stream* stream);
PAS_API void pas_status_reporter_dump_compact_bootstrap_free_heap(pas_stream* stream);
PAS_API void pas_status_reporter_dump_bootstrap_free_heap(pas_stream* stream);

PAS_API void pas_status_reporter_dump_bitfit_heap(pas_stream* stream, pas_bitfit_heap* heap);
PAS_API void pas_status_reporter_dump_segregated_heap(pas_stream* stream, pas_segregated_heap* heap);
PAS_API void pas_status_reporter_dump_heap(pas_stream* stream, pas_heap* heap);
PAS_API void pas_status_reporter_dump_all_heaps(pas_stream* stream);
PAS_API void pas_status_reporter_dump_all_shared_page_directories(pas_stream* stream);
PAS_API void pas_status_reporter_dump_all_heaps_non_utility_summaries(pas_stream* stream);
PAS_API void pas_status_reporter_dump_large_sharing_pool(pas_stream* stream);
PAS_API void pas_status_reporter_dump_utility_heap(pas_stream* stream);
PAS_API void pas_status_reporter_dump_total_fragmentation(pas_stream* stream);
PAS_API void pas_status_reporter_dump_view_stats(pas_stream* stream);
PAS_API void pas_status_reporter_dump_tier_up_rates(pas_stream* stream);
PAS_API void pas_status_reporter_dump_baseline_allocators(pas_stream* stream);
PAS_API void pas_status_reporter_dump_thread_local_caches(pas_stream* stream);
PAS_API void pas_status_reporter_dump_configuration(pas_stream* stream);
PAS_API void pas_status_reporter_dump_physical_page_sharing_pool(pas_stream* stream);
PAS_API void pas_status_reporter_dump_expendable_memories(pas_stream* stream);
PAS_API void pas_status_reporter_dump_everything(pas_stream* stream);

PAS_API void pas_status_reporter_start_if_necessary(void);

PAS_END_EXTERN_C;

#endif /* PAS_STATUS_REPORTER_H */

