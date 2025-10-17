/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 19, 2022.
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

#include "pas_segregated_page_config_kind_and_role.h"

PAS_BEGIN_EXTERN_C;

const char*
pas_segregated_page_config_kind_and_role_get_string(pas_segregated_page_config_kind_and_role kind_and_role)
{
    switch (kind_and_role) {
#define PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND(name, value) \
    case pas_segregated_page_config_kind_ ## name ## _and_shared_role: \
        return "shared_" #name; \
    case pas_segregated_page_config_kind_ ## name ## _and_exclusive_role: \
        return "exclusive_" #name;
#include "pas_segregated_page_config_kind.def"
#undef PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* LIBPAS_ENABLED */

