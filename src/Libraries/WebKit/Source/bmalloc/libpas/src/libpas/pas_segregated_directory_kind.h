/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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
#ifndef PAS_SEGREGATED_DIRECTORY_KIND_H
#define PAS_SEGREGATED_DIRECTORY_KIND_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

/* Segregated directories form a type hierarchy like so:
   
       - directory
           - size directory
           - shared page directory
   
   Only the types at the leaves are instantiable. The pas_segregated_directory_kind enum is for
   describing the instantiable leaves. */

enum pas_segregated_directory_kind {
    pas_segregated_size_directory_kind,
    pas_segregated_shared_page_directory_kind
};

typedef enum pas_segregated_directory_kind pas_segregated_directory_kind;

static inline const char* pas_segregated_directory_kind_get_string(
    pas_segregated_directory_kind kind)
{
    switch (kind) {
    case pas_segregated_size_directory_kind:
        return "segregated_size_directory";
    case pas_segregated_shared_page_directory_kind:
        return "segregated_shared_page_directory";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_SEGREGATED_DIRECTORY_KIND_H */

