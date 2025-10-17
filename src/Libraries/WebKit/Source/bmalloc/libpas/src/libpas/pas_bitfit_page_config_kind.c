/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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

#include "pas_bitfit_page_config_kind.h"

#include "pas_all_heap_configs.h"

PAS_BEGIN_EXTERN_C;

bool pas_small_bitfit_page_config_variant_is_enabled_override = PAS_USE_SMALL_BITFIT_OVERRIDE;
bool pas_medium_bitfit_page_config_variant_is_enabled_override = PAS_USE_MEDIUM_BITFIT_OVERRIDE;
bool pas_marge_bitfit_page_config_variant_is_enabled_override = PAS_USE_MARGE_BITFIT_OVERRIDE;

const char* pas_bitfit_page_config_kind_get_string(pas_bitfit_page_config_kind kind)
{
    switch (kind) {
#define PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND(name, value) \
    case pas_bitfit_page_config_kind_ ## name: \
        return #name;
#include "pas_bitfit_page_config_kind.def"
#undef PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND
    }
    PAS_ASSERT(!"Invalid kind");
    return NULL;
}

bool pas_bitfit_page_config_kind_for_each(
    pas_bitfit_page_config_kind_callback callback,
    void *arg)
{
#define PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND(name, value) \
    if (!callback(pas_bitfit_page_config_kind_ ## name, \
                  pas_page_base_config_get_bitfit((value).base.page_config_ptr), \
                  arg)) \
        return false;
#include "pas_bitfit_page_config_kind.def"
#undef PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND
    return true;    
}

const pas_page_base_config* pas_bitfit_page_config_kind_for_config_table[
    0
#define PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND(name, value) + 1
#include "pas_bitfit_page_config_kind.def"
#undef PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND
    ] = {
#define PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND(name, value) \
    (value).base.page_config_ptr,
#include "pas_bitfit_page_config_kind.def"
#undef PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND
};

PAS_END_EXTERN_C;

#endif /* LIBPAS_ENABLED */
