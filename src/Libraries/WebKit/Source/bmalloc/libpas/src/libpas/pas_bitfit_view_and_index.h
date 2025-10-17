/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 8, 2024.
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
#ifndef PAS_BITFIT_VIEW_AND_INDEX_H
#define PAS_BITFIT_VIEW_AND_INDEX_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_bitfit_view;
struct pas_bitfit_view_and_index;
typedef struct pas_bitfit_view pas_bitfit_view;
typedef struct pas_bitfit_view_and_index pas_bitfit_view_and_index;

struct pas_bitfit_view_and_index {
    pas_bitfit_view* view;
    size_t index;
};

static inline pas_bitfit_view_and_index pas_bitfit_view_and_index_create(pas_bitfit_view* view,
                                                                         size_t index)
{
    pas_bitfit_view_and_index result;
    result.view = view;
    result.index = index;
    return result;
}

static inline pas_bitfit_view_and_index pas_bitfit_view_and_index_create_empty(void)
{
    return pas_bitfit_view_and_index_create(NULL, 0);
}

PAS_END_EXTERN_C;

#endif /* PAS_BITFIT_VIEW_AND_INDEX_H */

