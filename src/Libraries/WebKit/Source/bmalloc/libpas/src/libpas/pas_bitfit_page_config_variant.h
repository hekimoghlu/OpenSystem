/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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
#ifndef PAS_BITFIT_PAGE_CONFIG_VARIANT_H
#define PAS_BITFIT_PAGE_CONFIG_VARIANT_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

/* Every bitfit heap has two page variants - small and medium. This tells us which. */
enum pas_bitfit_page_config_variant {
    pas_small_bitfit_page_config_variant,
    pas_medium_bitfit_page_config_variant,
    pas_marge_bitfit_page_config_variant
};

typedef enum pas_bitfit_page_config_variant pas_bitfit_page_config_variant;

#define PAS_NUM_BITFIT_PAGE_CONFIG_VARIANTS 3

#define PAS_EACH_BITFIT_PAGE_CONFIG_VARIANT_ASCENDING(variable) \
    variable = pas_small_bitfit_page_config_variant; \
    (unsigned)variable <= (unsigned)pas_marge_bitfit_page_config_variant; \
    variable = (pas_bitfit_page_config_variant)((unsigned)variable + 1)

#define PAS_EACH_BITFIT_PAGE_CONFIG_VARIANT_DESCENDING(variable) \
    variable = pas_marge_bitfit_page_config_variant; \
    (int)variable >= (int)pas_small_bitfit_page_config_variant; \
    variable = (pas_bitfit_page_config_variant)((unsigned)variable - 1)

static inline const char*
pas_bitfit_page_config_variant_get_string(pas_bitfit_page_config_variant variant)
{
    switch (variant) {
    case pas_small_bitfit_page_config_variant:
        return "small";
    case pas_medium_bitfit_page_config_variant:
        return "medium";
    case pas_marge_bitfit_page_config_variant:
        return "marge";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

static inline const char*
pas_bitfit_page_config_variant_get_capitalized_string(pas_bitfit_page_config_variant variant)
{
    switch (variant) {
    case pas_small_bitfit_page_config_variant:
        return "Small";
    case pas_medium_bitfit_page_config_variant:
        return "Medium";
    case pas_marge_bitfit_page_config_variant:
        return "Marge";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_BITFIT_PAGE_CONFIG_VARIANT_H */

