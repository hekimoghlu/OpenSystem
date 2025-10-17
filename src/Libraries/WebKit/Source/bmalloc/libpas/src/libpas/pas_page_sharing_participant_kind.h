/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
#ifndef PAS_PAGE_SHARING_PARTICIPANT_KIND_H
#define PAS_PAGE_SHARING_PARTICIPANT_KIND_H

#include "pas_segregated_directory_kind.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_page_sharing_participant_kind {
    pas_page_sharing_participant_null,
    pas_page_sharing_participant_segregated_shared_page_directory,
    pas_page_sharing_participant_segregated_size_directory,
    pas_page_sharing_participant_bitfit_directory,
    pas_page_sharing_participant_large_sharing_pool
};

typedef enum pas_page_sharing_participant_kind pas_page_sharing_participant_kind;

#define PAS_PAGE_SHARING_PARTICIPANT_KIND_MASK ((uintptr_t)7)

static inline pas_page_sharing_participant_kind
pas_page_sharing_participant_kind_select_for_segregated_directory(
    pas_segregated_directory_kind directory_kind)
{
    switch (directory_kind) {
    case pas_segregated_size_directory_kind:
        return pas_page_sharing_participant_segregated_size_directory;
    case pas_segregated_shared_page_directory_kind:
        return pas_page_sharing_participant_segregated_shared_page_directory;
    }
    PAS_ASSERT(!"Should not be reached");
    return pas_page_sharing_participant_null;
}

static inline const char*
pas_page_sharing_participant_kind_get_string(pas_page_sharing_participant_kind kind)
{
    switch (kind) {
    case pas_page_sharing_participant_null:
        return "null";
    case pas_page_sharing_participant_segregated_shared_page_directory:
        return "segregated_shared_page_directory";
    case pas_page_sharing_participant_segregated_size_directory:
        return "segregated_size_directory";
    case pas_page_sharing_participant_bitfit_directory:
        return "bitfit_directory";
    case pas_page_sharing_participant_large_sharing_pool:
        return "large_sharing_pool";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_PAGE_SHARING_PARTICIPANT_KIND_H */

