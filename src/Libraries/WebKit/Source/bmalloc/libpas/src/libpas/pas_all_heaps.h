/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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
#ifndef PAS_ALL_HEAPS_H
#define PAS_ALL_HEAPS_H

#include "pas_allocator_scavenge_action.h"
#include "pas_heap_summary.h"
#include "pas_lock.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

#ifndef pas_heap
#define pas_heap __pas_heap
#endif

struct pas_heap;
struct pas_heap_config;
struct pas_segregated_directory;
struct pas_segregated_heap;
typedef struct pas_heap pas_heap;
typedef struct pas_heap_config pas_heap_config;
typedef struct pas_segregated_directory pas_segregated_directory;
typedef struct pas_segregated_heap pas_segregated_heap;

PAS_API extern pas_heap* pas_all_heaps_first_heap;
PAS_API extern size_t pas_all_heaps_count;

/* Have to hold the heap lock to call this. */
PAS_API void pas_all_heaps_add_heap(pas_heap* heap);

typedef bool (*pas_all_heaps_for_each_heap_callback)(pas_heap* heap, void* arg);

PAS_API bool pas_all_heaps_for_each_static_heap(pas_all_heaps_for_each_heap_callback callback,
                                                void* arg);

PAS_API bool pas_all_heaps_for_each_dynamic_heap(pas_all_heaps_for_each_heap_callback callback,
                                                 void* arg);

/* Have to hold the heap lock to call this. 
   NOTE: This doesn't give you the utility heap. */
PAS_API bool pas_all_heaps_for_each_heap(pas_all_heaps_for_each_heap_callback callback,
                                         void* arg);

typedef bool (*pas_all_heaps_for_each_segregated_heap_callback)(
    pas_segregated_heap* heap, const pas_heap_config* heap_config, void* arg);

PAS_API bool pas_all_heaps_for_each_static_segregated_heap_not_part_of_a_heap(
    pas_all_heaps_for_each_segregated_heap_callback callback,
    void* arg);

PAS_API bool pas_all_heaps_for_each_static_segregated_heap(
    pas_all_heaps_for_each_segregated_heap_callback callback,
    void* arg);

/* Have to hold the heap lock to call this. */
PAS_API bool pas_all_heaps_for_each_segregated_heap(
    pas_all_heaps_for_each_segregated_heap_callback callback,
    void* arg);

PAS_API size_t pas_all_heaps_get_num_free_bytes(pas_lock_hold_mode heap_lock_hold_mode);

PAS_API void pas_all_heaps_reset_heap_ref(pas_lock_hold_mode heap_lock_hold_mode);

typedef bool (*pas_all_heaps_for_each_segregated_directory_callback)(
    pas_segregated_directory* directory, void* arg);

PAS_API bool pas_all_heaps_for_each_segregated_directory(
    pas_all_heaps_for_each_segregated_directory_callback callback,
    void* arg);

/* Run assertions that are unique to the case where we have synchronously scavenged and nobody has
   done any allocations or deallocations since then. It's hard to arrange to be in this state. It's
   probably limited to this workflow:
   
   Step #1: Somehow arrange for all other threads to pause, but not inside libpas code, or any code that
            libpas code transitively uses. They have to stay paused for the rest of this workflow.
   Step #2: pas_scavenger_run_synchronously_now()
   Step #3: pas_all_heaps_verify_in_steady_state()
   
   This should probably only be used by tests. */
PAS_API void pas_all_heaps_verify_in_steady_state(void);

PAS_API pas_heap_summary pas_all_heaps_compute_total_non_utility_segregated_summary(void);
PAS_API pas_heap_summary pas_all_heaps_compute_total_non_utility_bitfit_summary(void);
PAS_API pas_heap_summary pas_all_heaps_compute_total_non_utility_large_summary(void);
PAS_API pas_heap_summary pas_all_heaps_compute_total_non_utility_summary(void);

PAS_END_EXTERN_C;

#endif /* PAS_ALL_HEAPS_H */

