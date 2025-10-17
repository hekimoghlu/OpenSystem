/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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
#ifndef PAS_PAGE_BASE_CONFIG_INLINES_H
#define PAS_PAGE_BASE_CONFIG_INLINES_H

#include "pas_segregated_page_config.h"

PAS_BEGIN_EXTERN_C;

static inline bool pas_page_base_config_is_utility(const pas_page_base_config* config)
{
    return pas_page_base_config_is_segregated(*config)
        && pas_segregated_page_config_is_utility(*pas_page_base_config_get_segregated(config));
}

PAS_END_EXTERN_C;

#endif /* PAS_PAGE_BASE_CONFIG_INLINES_H */

