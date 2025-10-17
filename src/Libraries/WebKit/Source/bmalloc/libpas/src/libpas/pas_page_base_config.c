/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_page_base_config.h"

#include "pas_bitfit_page_config.h"
#include "pas_segregated_page_config.h"

PAS_BEGIN_EXTERN_C;

const char* pas_page_base_config_get_kind_string(const pas_page_base_config* config)
{
    switch (config->page_config_kind) {
    case pas_page_config_kind_segregated:
        return pas_segregated_page_config_kind_get_string(pas_page_base_config_get_segregated(config)->kind);
    case pas_page_config_kind_bitfit:
        return pas_bitfit_page_config_kind_get_string(pas_page_base_config_get_bitfit(config)->kind);
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* LIBPAS_ENABLED */

