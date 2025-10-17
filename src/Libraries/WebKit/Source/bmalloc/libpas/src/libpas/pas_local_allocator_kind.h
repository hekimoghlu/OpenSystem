/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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
#ifndef PAS_LOCAL_ALLOCATOR_KIND
#define PAS_LOCAL_ALLOCATOR_KIND

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_local_allocator_kind {
    pas_local_allocator_decommitted_kind,
    pas_local_allocator_stopped_allocator_kind,
    pas_local_allocator_allocator_kind,
    pas_local_allocator_stopped_view_cache_kind,
    pas_local_allocator_view_cache_kind
};

typedef enum pas_local_allocator_kind pas_local_allocator_kind;

static inline const char* pas_local_allocator_kind_get_string(pas_local_allocator_kind kind)
{
    switch (kind) {
    case pas_local_allocator_decommitted_kind:
        return "decommitted";
    case pas_local_allocator_stopped_allocator_kind:
        return "stopped_allocator";
    case pas_local_allocator_allocator_kind:
        return "allocator";
    case pas_local_allocator_stopped_view_cache_kind:
        return "stopped_view_cache";
    case pas_local_allocator_view_cache_kind:
        return "view_cache";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

static inline bool pas_local_allocator_kind_is_stopped(pas_local_allocator_kind kind)
{
    switch (kind) { 
    case pas_local_allocator_decommitted_kind:
    case pas_local_allocator_stopped_allocator_kind:
    case pas_local_allocator_stopped_view_cache_kind:
        return true;
    case pas_local_allocator_allocator_kind:
    case pas_local_allocator_view_cache_kind:
        return false;
    }
    PAS_ASSERT(!"Should not be reached");
    return false;
}

PAS_END_EXTERN_C;

#endif /* PAS_LOCAL_ALLOCATOR_KIND */

