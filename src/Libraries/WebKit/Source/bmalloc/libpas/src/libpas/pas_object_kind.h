/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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
#ifndef PAS_OBJECT_KIND_H
#define PAS_OBJECT_KIND_H

#include "pas_bitfit_page_config_variant.h"
#include "pas_page_kind.h"
#include "pas_segregated_page_config_variant.h"

PAS_BEGIN_EXTERN_C;

enum pas_object_kind {
    pas_not_an_object_kind,
    pas_small_segregated_object_kind,
    pas_medium_segregated_object_kind,
    pas_small_bitfit_object_kind,
    pas_medium_bitfit_object_kind,
    pas_marge_bitfit_object_kind,
    pas_large_object_kind
};

typedef enum pas_object_kind pas_object_kind;

static inline const char* pas_object_kind_get_string(pas_object_kind object_kind)
{
    switch (object_kind) {
    case pas_small_segregated_object_kind:
        return "small_segregated";
    case pas_medium_segregated_object_kind:
        return "medium_segregated";
    case pas_small_bitfit_object_kind:
        return "small_bitfit";
    case pas_medium_bitfit_object_kind:
        return "medium_bitfit";
    case pas_marge_bitfit_object_kind:
        return "marge_bitfit";
    case pas_large_object_kind:
        return "large";
    case pas_not_an_object_kind:
        return "not_an_object";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

static inline pas_object_kind
pas_object_kind_for_segregated_variant(pas_segregated_page_config_variant variant)
{
    switch (variant) {
    case pas_small_segregated_page_config_variant:
        return pas_small_segregated_object_kind;
    case pas_medium_segregated_page_config_variant:
        return pas_medium_segregated_object_kind;
    }
    PAS_ASSERT(!"Should not be reached");
    return pas_not_an_object_kind;
}

static inline pas_object_kind
pas_object_kind_for_bitfit_variant(pas_bitfit_page_config_variant variant)
{
    switch (variant) {
    case pas_small_bitfit_page_config_variant:
        return pas_small_bitfit_object_kind;
    case pas_medium_bitfit_page_config_variant:
        return pas_medium_bitfit_object_kind;
    case pas_marge_bitfit_page_config_variant:
        return pas_marge_bitfit_object_kind;
    }
    PAS_ASSERT(!"Should not be reached");
    return pas_not_an_object_kind;
}

static inline pas_object_kind pas_object_kind_for_page_kind(pas_page_kind page_kind)
{
    switch (page_kind) {
    case pas_small_exclusive_segregated_page_kind:
    case pas_small_shared_segregated_page_kind:
        return pas_small_segregated_object_kind;
    case pas_medium_exclusive_segregated_page_kind:
    case pas_medium_shared_segregated_page_kind:
        return pas_medium_segregated_object_kind;
    case pas_small_bitfit_page_kind:
        return pas_small_bitfit_object_kind;
    case pas_medium_bitfit_page_kind:
        return pas_medium_bitfit_object_kind;
    case pas_marge_bitfit_page_kind:
        return pas_marge_bitfit_object_kind;
    }
    PAS_ASSERT(!"Should not be reached");
    return pas_not_an_object_kind;
}

PAS_END_EXTERN_C;

#endif /* PAS_OBJECT_KIND_H */

