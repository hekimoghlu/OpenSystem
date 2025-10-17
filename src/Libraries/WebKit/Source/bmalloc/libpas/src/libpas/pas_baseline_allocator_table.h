/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 4, 2023.
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
#ifndef PAS_BASELINE_ALLOCATOR_TABLE_H
#define PAS_BASELINE_ALLOCATOR_TABLE_H

#include "pas_baseline_allocator.h"

PAS_BEGIN_EXTERN_C;

PAS_API extern pas_baseline_allocator* pas_baseline_allocator_table;
PAS_API extern uint64_t pas_num_baseline_allocator_evictions;
PAS_API extern unsigned pas_baseline_allocator_table_bound;

PAS_API void pas_baseline_allocator_table_initialize_if_necessary(void);

PAS_API unsigned pas_baseline_allocator_table_get_random_index(void);

PAS_API bool pas_baseline_allocator_table_for_all(pas_allocator_scavenge_action action);

PAS_END_EXTERN_C;

#endif /* PAS_BASELINE_ALLOCATOR_TABLE_H */

