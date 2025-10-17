/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_heap_config_kind.h"

#include "pas_all_heap_configs.h"
#include "pas_heap_lock.h"

const char* pas_heap_config_kind_get_string(pas_heap_config_kind kind)
{
    switch (kind) {
#define PAS_DEFINE_HEAP_CONFIG_KIND(name, value) \
    case pas_heap_config_kind_ ## name: \
        return #name;
#include "pas_heap_config_kind.def"
#undef PAS_DEFINE_HEAP_CONFIG_KIND
    }
    PAS_ASSERT(!"Invalid kind");
    return NULL;
}

PAS_BEGIN_EXTERN_C;

bool pas_heap_config_kind_for_each(
    pas_heap_config_kind_callback callback,
    void *arg)
{
#define PAS_DEFINE_HEAP_CONFIG_KIND(name, value) \
    if (!callback(pas_heap_config_kind_ ## name, \
                  (value).config_ptr, \
                  arg)) \
        return false;
#include "pas_heap_config_kind.def"
#undef PAS_DEFINE_HEAP_CONFIG_KIND
    return true;    
}

const pas_heap_config* pas_heap_config_kind_for_config_table[pas_heap_config_kind_num_kinds] = {
#define PAS_DEFINE_HEAP_CONFIG_KIND(name, value) \
    (value).config_ptr,
#include "pas_heap_config_kind.def"
#undef PAS_DEFINE_HEAP_CONFIG_KIND
};

unsigned pas_heap_config_kind_is_active_bitvector[
    PAS_BITVECTOR_NUM_WORDS(pas_heap_config_kind_num_kinds)];

bool pas_heap_config_kind_set_active(pas_heap_config_kind kind)
{
    PAS_TESTING_ASSERT((unsigned)kind < (unsigned)pas_heap_config_kind_num_kinds);
    pas_heap_lock_assert_held();
    if (pas_bitvector_get(pas_heap_config_kind_is_active_bitvector, (size_t)kind))
        return false;
    pas_bitvector_set(pas_heap_config_kind_is_active_bitvector, (size_t)kind, true);
    return true;
}

PAS_END_EXTERN_C;

#endif /* LIBPAS_ENABLED */
