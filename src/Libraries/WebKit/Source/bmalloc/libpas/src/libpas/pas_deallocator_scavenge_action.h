/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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
#ifndef PAS_DEALLOCATOR_SCAVENGE_ACTION_H
#define PAS_DEALLOCATOR_SCAVENGE_ACTION_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_deallocator_scavenge_action {
    pas_deallocator_scavenge_no_action,
    pas_deallocator_scavenge_flush_log_if_clean_action,
    pas_deallocator_scavenge_flush_log_action
};

typedef enum pas_deallocator_scavenge_action pas_deallocator_scavenge_action;

static inline const char*
pas_deallocator_scavenge_action_get_string(pas_deallocator_scavenge_action action)
{
    switch (action)
    {
    case pas_deallocator_scavenge_no_action:
        return "no_action";
    case pas_deallocator_scavenge_flush_log_if_clean_action:
        return "flush_log_if_clean_action";
    case pas_deallocator_scavenge_flush_log_action:
        return "flush_log_action";
    }
    
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_DEALLOCATOR_SCAVENGE_ACTION_H */

