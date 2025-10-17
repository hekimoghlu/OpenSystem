/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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
#ifndef PAS_BITFIT_PAGE_CONFIG_KIND_H
#define PAS_BITFIT_PAGE_CONFIG_KIND_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_page_base_config;
struct pas_bitfit_page_config;
typedef struct pas_page_base_config pas_page_base_config;
typedef struct pas_bitfit_page_config pas_bitfit_page_config;

enum pas_bitfit_page_config_kind {
#define PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND(name, value) \
    pas_bitfit_page_config_kind_ ## name,
#include "pas_bitfit_page_config_kind.def"
#undef PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND
};

typedef enum pas_bitfit_page_config_kind pas_bitfit_page_config_kind;

PAS_API const char* pas_bitfit_page_config_kind_get_string(pas_bitfit_page_config_kind kind);

typedef bool (*pas_bitfit_page_config_kind_callback)(pas_bitfit_page_config_kind kind,
                                                         const pas_bitfit_page_config* config,
                                                         void* arg);

PAS_API bool pas_bitfit_page_config_kind_for_each(
    pas_bitfit_page_config_kind_callback callback,
    void *arg);

PAS_API extern const pas_page_base_config* pas_bitfit_page_config_kind_for_config_table[];

static inline const pas_bitfit_page_config* pas_bitfit_page_config_kind_get_config(
    pas_bitfit_page_config_kind kind)
{
    return (const pas_bitfit_page_config*)pas_bitfit_page_config_kind_for_config_table[kind];
}

PAS_END_EXTERN_C;

#endif /* PAS_BITFIT_PAGE_CONFIG_KIND_H */

