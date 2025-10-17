/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 21, 2024.
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
#ifndef PAS_TREE_DIRECTION_H
#define PAS_TREE_DIRECTION_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_tree_direction {
    pas_tree_direction_left,
    pas_tree_direction_right
};

typedef enum pas_tree_direction pas_tree_direction;

static inline const char* pas_tree_direction_get_string(pas_tree_direction direction)
{
    switch (direction) {
    case pas_tree_direction_left:
        return "left";
    case pas_tree_direction_right:
        return "right";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

static inline pas_tree_direction pas_tree_direction_invert(pas_tree_direction direction)
{
    switch (direction) {
    case pas_tree_direction_left:
        return pas_tree_direction_right;
    case pas_tree_direction_right:
        return pas_tree_direction_left;
    }
    PAS_ASSERT(!"Should not be reached");
    return pas_tree_direction_left;
}

static inline int
pas_tree_direction_invert_comparison_result_if_right(
    pas_tree_direction direction,
    int comparison_result)
{
    switch (direction) {
    case pas_tree_direction_left:
        return comparison_result;
    case pas_tree_direction_right:
        return -comparison_result;
    }
    PAS_ASSERT(!"Should not be reached");
    return 0;
}

PAS_END_EXTERN_C;

#endif /* PAS_TREE_DIRECTION_H */

