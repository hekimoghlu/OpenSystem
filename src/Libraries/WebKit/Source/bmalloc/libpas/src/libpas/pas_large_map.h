/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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
#ifndef PAS_LARGE_MAP_H
#define PAS_LARGE_MAP_H

#include "pas_bootstrap_free_heap.h"
#include "pas_config.h"
#include "pas_first_level_tiny_large_map_entry.h"
#include "pas_hashtable.h"
#include "pas_large_map_entry.h"
#include "pas_small_large_map_entry.h"
#include "pas_tiny_large_map_entry.h"
#include "pas_utility_heap.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_large_heap;
typedef struct pas_large_heap pas_large_heap;

PAS_CREATE_HASHTABLE(pas_large_map_hashtable,
                     pas_large_map_entry,
                     pas_large_map_key);

PAS_API extern pas_large_map_hashtable pas_large_map_hashtable_instance;
PAS_API extern pas_large_map_hashtable_in_flux_stash pas_large_map_hashtable_instance_in_flux_stash;

PAS_CREATE_HASHTABLE(pas_small_large_map_hashtable,
                     pas_small_large_map_entry,
                     pas_large_map_key);

PAS_API extern pas_small_large_map_hashtable pas_small_large_map_hashtable_instance;
PAS_API extern pas_small_large_map_hashtable_in_flux_stash pas_small_large_map_hashtable_instance_in_flux_stash;

PAS_CREATE_HASHTABLE(pas_tiny_large_map_hashtable,
                     pas_first_level_tiny_large_map_entry,
                     pas_first_level_tiny_large_map_key);

PAS_API extern pas_tiny_large_map_hashtable pas_tiny_large_map_hashtable_instance;
PAS_API extern pas_tiny_large_map_hashtable_in_flux_stash pas_tiny_large_map_hashtable_instance_in_flux_stash;
PAS_API extern pas_tiny_large_map_second_level_hashtable_in_flux_stash pas_tiny_large_map_second_level_hashtable_in_flux_stash_instance;

PAS_API pas_large_map_entry pas_large_map_find(uintptr_t begin);

PAS_API void pas_large_map_add(pas_large_map_entry entry);
PAS_API pas_large_map_entry pas_large_map_take(uintptr_t begin);

typedef bool (*pas_large_map_for_each_entry_callback)(pas_large_map_entry entry,
                                                      void* arg);

PAS_API bool pas_large_map_for_each_entry(pas_large_map_for_each_entry_callback callback,
                                          void *arg);

PAS_END_EXTERN_C;

#endif /* PAS_LARGE_MAP_H */

