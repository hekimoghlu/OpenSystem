/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 13, 2023.
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
#ifndef PAS_SEGREGATED_PAGE_AND_CONFIG_H
#define PAS_SEGREGATED_PAGE_AND_CONFIG_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_segregated_page;
struct pas_segregated_page_and_config;
struct pas_segregated_page_config;
typedef struct pas_segregated_page pas_segregated_page;
typedef struct pas_segregated_page_and_config pas_segregated_page_and_config;
typedef struct pas_segregated_page_config pas_segregated_page_config;

struct pas_segregated_page_and_config {
    pas_segregated_page* page;
    const pas_segregated_page_config* config;
};

static inline pas_segregated_page_and_config
pas_segregated_page_and_config_create(pas_segregated_page* page,
                                      const pas_segregated_page_config* config)
{
    pas_segregated_page_and_config result;
    PAS_ASSERT(!!page == !!config);
    result.page = page;
    result.config = config;
    return result;
}

static inline pas_segregated_page_and_config
pas_segregated_page_and_config_create_empty(void)
{
    return pas_segregated_page_and_config_create(NULL, NULL);
}

static inline bool pas_segregated_page_and_config_is_empty(
    pas_segregated_page_and_config page_and_config)
{
    PAS_ASSERT(!!page_and_config.page == !!page_and_config.config);
    return !page_and_config.page;
}

PAS_END_EXTERN_C;

#endif /* PAS_SEGREGATED_PAGE_AND_CONFIG_H */


