/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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
#ifndef PAS_HEAP_CONFIG_KIND_H
#define PAS_HEAP_CONFIG_KIND_H

#include "pas_bitvector.h"

PAS_BEGIN_EXTERN_C;

struct pas_heap_config;
typedef struct pas_heap_config pas_heap_config;

enum pas_heap_config_kind {
#define PAS_DEFINE_HEAP_CONFIG_KIND(name, value) \
    pas_heap_config_kind_ ## name,
#include "pas_heap_config_kind.def"
#undef PAS_DEFINE_HEAP_CONFIG_KIND
};

typedef enum pas_heap_config_kind pas_heap_config_kind;

enum { pas_heap_config_kind_num_kinds =
    0
#define PAS_DEFINE_HEAP_CONFIG_KIND(name, value) \
    + 1
#include "pas_heap_config_kind.def"
#undef PAS_DEFINE_HEAP_CONFIG_KIND
};

PAS_API const char* pas_heap_config_kind_get_string(pas_heap_config_kind kind);

typedef bool (*pas_heap_config_kind_callback)(pas_heap_config_kind kind,
                                              const pas_heap_config* config,
                                              void* arg);

PAS_API bool pas_heap_config_kind_for_each(
    pas_heap_config_kind_callback callback,
    void *arg);

#define PAS_EACH_HEAP_CONFIG_KIND(kind) \
    kind = (pas_heap_config_kind)0; \
    (unsigned)kind < (unsigned)pas_heap_config_kind_num_kinds; \
    kind = (pas_heap_config_kind)((unsigned)kind + 1)

PAS_API extern const pas_heap_config* pas_heap_config_kind_for_config_table[];

static inline const pas_heap_config* pas_heap_config_kind_get_config(pas_heap_config_kind kind)
{
    PAS_TESTING_ASSERT((unsigned)kind < (unsigned)pas_heap_config_kind_num_kinds);
    return pas_heap_config_kind_for_config_table[kind];
}

PAS_API extern unsigned pas_heap_config_kind_is_active_bitvector[];

static inline bool pas_heap_config_kind_is_active(pas_heap_config_kind kind)
{
    PAS_TESTING_ASSERT((unsigned)kind < (unsigned)pas_heap_config_kind_num_kinds);
    return pas_bitvector_get(pas_heap_config_kind_is_active_bitvector, (size_t)kind);
}

/* Returns true if we did set the bit. Must be called with heap lock held. */
PAS_API bool pas_heap_config_kind_set_active(pas_heap_config_kind kind);

PAS_END_EXTERN_C;

#endif /* PAS_HEAP_CONFIG_KIND_H */

