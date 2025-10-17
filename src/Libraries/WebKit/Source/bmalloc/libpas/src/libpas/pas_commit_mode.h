/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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
#ifndef PAS_COMMIT_MODE_H
#define PAS_COMMIT_MODE_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

/* pas_commit_mode is a named "is committed" boolean. */

enum pas_commit_mode {
    pas_decommitted,
    pas_committed
};

typedef enum pas_commit_mode pas_commit_mode;

static inline const char* pas_commit_mode_get_string(pas_commit_mode mode)
{
    switch (mode) {
    case pas_decommitted:
        return "decommitted";
    case pas_committed:
        return "committed";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_COMMIT_MODE_H */

