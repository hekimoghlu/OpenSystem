/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
#ifndef PAS_SEGREGATED_PAGE_CONFIG_KIND_AND_ROLE_H
#define PAS_SEGREGATED_PAGE_CONFIG_KIND_AND_ROLE_H

#include "pas_segregated_page_config_kind.h"
#include "pas_segregated_page_role.h"

PAS_BEGIN_EXTERN_C;

enum pas_segregated_page_config_kind_and_role {
#define PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND(name, value) \
    pas_segregated_page_config_kind_ ## name ## _and_shared_role, \
    pas_segregated_page_config_kind_ ## name ## _and_exclusive_role,
#include "pas_segregated_page_config_kind.def"
#undef PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND
};

typedef enum pas_segregated_page_config_kind_and_role pas_segregated_page_config_kind_and_role;

#define PAS_SEGREGATED_PAGE_CONFIG_KIND_AND_ROLE_NUM_BITS 6ull
#define PAS_SEGREGATED_PAGE_CONFIG_KIND_AND_ROLE_SHIFT 48ull
#define PAS_SEGREGATED_PAGE_CONFIG_KIND_AND_ROLE_MASK \
    (((1u << PAS_SEGREGATED_PAGE_CONFIG_KIND_AND_ROLE_NUM_BITS) - 1ull) << PAS_SEGREGATED_PAGE_CONFIG_KIND_AND_ROLE_SHIFT)

#if PAS_COMPILER(CLANG)
#define PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND(name, value) \
    _Static_assert(pas_segregated_page_config_kind_ ## name ## _and_shared_role \
                   < (1u << PAS_SEGREGATED_PAGE_CONFIG_KIND_AND_ROLE_NUM_BITS), \
                   "Kind-and-role doesn't fit in kind-and-role bits"); \
    _Static_assert(pas_segregated_page_config_kind_ ## name ## _and_exclusive_role \
                   < (1u << PAS_SEGREGATED_PAGE_CONFIG_KIND_AND_ROLE_NUM_BITS), \
                   "Kind-and-role doesn't fit in kind-and-role bits");
#include "pas_segregated_page_config_kind.def"
#undef PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND
#endif /* PAS_COMPILER(CLANG) */

PAS_API const char*
pas_segregated_page_config_kind_and_role_get_string(pas_segregated_page_config_kind_and_role kind_and_role);

static inline pas_segregated_page_config_kind_and_role pas_segregated_page_config_kind_and_role_create(
    pas_segregated_page_config_kind kind,
    pas_segregated_page_role role)
{
    pas_segregated_page_config_kind_and_role result;
    result = (pas_segregated_page_config_kind_and_role)(((unsigned)kind << 1) | (unsigned)role);
    if (PAS_ENABLE_TESTING) {
        switch (kind) {
#define PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND(name, value) \
        case pas_segregated_page_config_kind_ ## name: \
            switch (role) { \
            case pas_segregated_page_shared_role: \
                PAS_ASSERT(result == pas_segregated_page_config_kind_ ## name ## _and_shared_role); \
                break; \
            case pas_segregated_page_exclusive_role: \
                PAS_ASSERT(result == pas_segregated_page_config_kind_ ## name ## _and_exclusive_role); \
                break; \
            } \
            break;
#include "pas_segregated_page_config_kind.def"
#undef PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND
        }
    }
    return result;
}

PAS_END_EXTERN_C;

#endif /* PAS_SEGREGATED_PAGE_CONFIG_KIND_AND_ROLE_H */

