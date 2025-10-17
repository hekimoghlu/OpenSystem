/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
#ifndef PAS_BIASING_DIRECTORY_KIND_H
#define PAS_BIASING_DIRECTORY_KIND_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_biasing_directory_kind {
    pas_biasing_directory_segregated_kind,
    pas_biasing_directory_bitfit_kind,
};

typedef enum pas_biasing_directory_kind pas_biasing_directory_kind;

static inline const char* pas_biasing_directory_kind_get_string(pas_biasing_directory_kind kind)
{
    switch (kind) {
    case pas_biasing_directory_segregated_kind:
        return "segregated_biasing_directory";
    case pas_biasing_directory_bitfit_kind:
        return "bitfit_biasing_directory";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_BIASING_DIRECTORY_KIND_H */

