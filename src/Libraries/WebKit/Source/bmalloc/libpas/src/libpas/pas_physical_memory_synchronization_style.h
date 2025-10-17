/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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
#ifndef PAS_PHYSICAL_MEMORY_SYNCHRONIZATION_STYLE_H
#define PAS_PHYSICAL_MEMORY_SYNCHRONIZATION_STYLE_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_physical_memory_synchronization_style {
    pas_physical_memory_is_locked_by_heap_lock,
    pas_physical_memory_is_locked_by_virtual_range_common_lock
};

typedef enum pas_physical_memory_synchronization_style pas_physical_memory_synchronization_style;

static inline const char* pas_physical_memory_synchronization_style_get_string(
    pas_physical_memory_synchronization_style style)
{
    switch (style) {
    case pas_physical_memory_is_locked_by_heap_lock:
        return "locked_by_heap_lock";
    case pas_physical_memory_is_locked_by_virtual_range_common_lock:
        return "locked_by_virtual_range_common_lock";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_PHYSICAL_MEMORY_SYNCHRONIZATION_STYLE_H */

