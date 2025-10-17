/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 7, 2023.
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
#ifndef PAS_BITFIT_PAGE_CONFIG_UTILS_H
#define PAS_BITFIT_PAGE_CONFIG_UTILS_H

#include "pas_config.h"
#include "pas_page_base_config_utils.h"
#include "pas_bitfit_page.h"
#include "pas_bitfit_page_config.h"
#include "pas_bitfit_page_inlines.h"

PAS_BEGIN_EXTERN_C;

#define PAS_BASIC_BITFIT_PAGE_CONFIG_FORWARD_DECLARATIONS(name) \
    PAS_BITFIT_PAGE_CONFIG_SPECIALIZATION_DECLARATIONS(name ## _page_config); \
    PAS_BASIC_PAGE_BASE_CONFIG_FORWARD_DECLARATIONS(name)

typedef struct {
    pas_page_header_placement_mode header_placement_mode;
    pas_page_header_table* header_table;
} pas_basic_bitfit_page_config_declarations_arguments;

#define PAS_BASIC_BITFIT_PAGE_CONFIG_DECLARATIONS(name, config_value, ...) \
    PAS_BASIC_PAGE_BASE_CONFIG_DECLARATIONS( \
        name, (config_value).base, __VA_ARGS__)

PAS_END_EXTERN_C;

#endif /* PAS_BITFIT_PAGE_CONFIG_UTILS_H */

