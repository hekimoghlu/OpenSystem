/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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
#ifndef PAS_ZERO_MODE_H
#define PAS_ZERO_MODE_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_zero_mode {
    pas_zero_mode_may_have_non_zero,
    pas_zero_mode_is_all_zero
};

typedef enum pas_zero_mode pas_zero_mode;

static inline const char* pas_zero_mode_get_string(pas_zero_mode mode)
{
    switch (mode) {
    case pas_zero_mode_may_have_non_zero:
        return "may_have_non_zero";
    case pas_zero_mode_is_all_zero:
        return "is_all_zero";
    }
    PAS_ASSERT(!"Invalid mode");
    return NULL;
}

static inline void pas_zero_mode_validate(pas_zero_mode mode)
{
    switch (mode) {
    case pas_zero_mode_may_have_non_zero:
    case pas_zero_mode_is_all_zero:
        return;
    }
    PAS_ASSERT(!"Invalid mode");
}

static inline pas_zero_mode pas_zero_mode_merge(pas_zero_mode left, pas_zero_mode right)
{
    switch (left) {
    case pas_zero_mode_may_have_non_zero:
        return pas_zero_mode_may_have_non_zero;
    case pas_zero_mode_is_all_zero:
        switch (right) {
        case pas_zero_mode_may_have_non_zero:
            return pas_zero_mode_may_have_non_zero;
        case pas_zero_mode_is_all_zero:
            return pas_zero_mode_is_all_zero;
        }
        PAS_ASSERT(!"Invalid right mode");
        return pas_zero_mode_may_have_non_zero;
    }
    PAS_ASSERT(!"Invalid left mode");
    return pas_zero_mode_may_have_non_zero;
}

PAS_END_EXTERN_C;

#endif /* PAS_ZERO_MODE_H */


