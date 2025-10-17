/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 25, 2023.
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
#ifndef PAS_MUTATION_COUNT_H
#define PAS_MUTATION_COUNT_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_mutation_count;
typedef struct pas_mutation_count pas_mutation_count;

struct pas_mutation_count {
    uintptr_t count;
};

#define PAS_MUTATION_COUNT_INITIALIZER ((pas_mutation_count){ .count = 0 })

#define PAS_MUTATION_COUNT_MUTATING_BIT ((uintptr_t)1)

/* Always call the start_mutating/stop_mutating functions while holding a lock. */

static inline void pas_mutation_count_start_mutating(pas_mutation_count* mutation_count)
{
    pas_compiler_fence();
    PAS_ASSERT(!(mutation_count->count & PAS_MUTATION_COUNT_MUTATING_BIT));
    mutation_count->count++;
    PAS_ASSERT(mutation_count->count & PAS_MUTATION_COUNT_MUTATING_BIT);
    pas_fence();
}

static inline void pas_mutation_count_stop_mutating(pas_mutation_count* mutation_count)
{
    pas_fence();
    PAS_ASSERT(mutation_count->count & PAS_MUTATION_COUNT_MUTATING_BIT);
    mutation_count->count++;
    PAS_ASSERT(!(mutation_count->count & PAS_MUTATION_COUNT_MUTATING_BIT));
    pas_compiler_fence();
}

static inline bool pas_mutation_count_is_mutating(pas_mutation_count saved_mutation_count)
{
    return saved_mutation_count.count & PAS_MUTATION_COUNT_MUTATING_BIT;
}

static inline bool pas_mutation_count_matches_with_dependency(pas_mutation_count* mutation_count,
                                                              pas_mutation_count saved_mutation_count,
                                                              uintptr_t dependency)
{
    pas_compiler_fence();
    return mutation_count[pas_depend(dependency)].count == saved_mutation_count.count;
}

static inline unsigned pas_mutation_count_depend(pas_mutation_count saved_mutation_count)
{
    return pas_depend((unsigned long)saved_mutation_count.count);
}

PAS_END_EXTERN_C;

#endif /* PAS_MUTATION_COUNT_H */

