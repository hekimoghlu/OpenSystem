/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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
#ifndef PAS_FREE_GRANULES_H
#define PAS_FREE_GRANULES_H

#include "pas_bitvector.h"
#include "pas_config.h"
#include "pas_lock.h"
#include "pas_page_granule_use_count.h"

PAS_BEGIN_EXTERN_C;

struct pas_deferred_decommit_log;
struct pas_free_granules;
struct pas_page_base;
struct pas_page_base_config;
typedef struct pas_deferred_decommit_log pas_deferred_decommit_log;
typedef struct pas_free_granules pas_free_granules;
typedef struct pas_page_base pas_page_base;
typedef struct pas_page_base_config pas_page_base_config;

struct pas_free_granules {
    unsigned free_granules[PAS_BITVECTOR_NUM_WORDS(PAS_MAX_GRANULES)];
    size_t num_free_granules;
    size_t num_already_decommitted_granules;
};

/* This must be done under the page's lock (ownership lock for most pages or the page's biased
   lock for segregated exclusive). */
PAS_API void pas_free_granules_compute_and_mark_decommitted(pas_free_granules* free_granules,
                                                            pas_page_granule_use_count* use_counts,
                                                            size_t num_granules);

PAS_API void pas_free_granules_unmark_decommitted(pas_free_granules* free_granules,
                                                  pas_page_granule_use_count* use_count,
                                                  size_t num_granules);

static inline bool pas_free_granules_is_free(pas_free_granules* free_granules,
                                             size_t index)
{
    PAS_ASSERT(index < PAS_MAX_GRANULES);
    return pas_bitvector_get(free_granules->free_granules, index);
}

/* It's the caller's responsibility to tell the log about what locks to acquire. */
PAS_API void pas_free_granules_decommit_after_locking_range(pas_free_granules* free_granules,
                                                            pas_page_base* page,
                                                            pas_deferred_decommit_log* log,
                                                            pas_lock* commit_lock,
                                                            const pas_page_base_config* page_config,
                                                            pas_lock_hold_mode heap_lock_hold_mode);

PAS_END_EXTERN_C;

#endif /* PAS_FREE_GRANULES_H */

