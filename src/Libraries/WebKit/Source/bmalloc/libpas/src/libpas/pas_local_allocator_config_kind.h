/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 30, 2023.
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
#ifndef PAS_LOCAL_ALLOCATOR_CONFIG_KIND_H
#define PAS_LOCAL_ALLOCATOR_CONFIG_KIND_H

#include "pas_bitfit_page_config_kind.h"
#include "pas_segregated_page_config_kind.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_local_allocator_config_kind {
    pas_local_allocator_config_kind_null,
    pas_local_allocator_config_kind_unselected,
#define PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND(name, value) \
    pas_local_allocator_config_kind_normal_ ## name, \
    pas_local_allocator_config_kind_primordial_partial_ ## name,
#include "pas_segregated_page_config_kind.def"
#undef PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND
#define PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND(name, value) \
    pas_local_allocator_config_kind_bitfit_ ## name,
#include "pas_bitfit_page_config_kind.def"
#undef PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND
};

typedef enum pas_local_allocator_config_kind pas_local_allocator_config_kind;

static PAS_ALWAYS_INLINE bool
pas_local_allocator_config_kind_is_primordial_partial(pas_local_allocator_config_kind kind)
{
    switch (kind) {
#define PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND(name, value) \
    case pas_local_allocator_config_kind_primordial_partial_ ## name: \
        return true;
#include "pas_segregated_page_config_kind.def"
#undef PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND
    default:
        return false;
    }
}

static PAS_ALWAYS_INLINE bool
pas_local_allocator_config_kind_is_bitfit(pas_local_allocator_config_kind kind)
{
    switch (kind) {
#define PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND(name, value) \
    case pas_local_allocator_config_kind_bitfit_ ## name: \
        return true;
#include "pas_bitfit_page_config_kind.def"
#undef PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND
    default:
        return false;
    }
}

static PAS_ALWAYS_INLINE pas_local_allocator_config_kind
pas_local_allocator_config_kind_create_normal(pas_segregated_page_config_kind kind)
{
    switch (kind) {
#define PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND(name, value) \
    case pas_segregated_page_config_kind_ ## name: \
        return pas_local_allocator_config_kind_normal_ ## name;
#include "pas_segregated_page_config_kind.def"
#undef PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND
    }
    PAS_ASSERT(!"Should not be reached");
    return (pas_local_allocator_config_kind)0;
}

static PAS_ALWAYS_INLINE pas_local_allocator_config_kind
pas_local_allocator_config_kind_create_primordial_partial(pas_segregated_page_config_kind kind)
{
    switch (kind) {
#define PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND(name, value) \
    case pas_segregated_page_config_kind_ ## name: \
        return pas_local_allocator_config_kind_primordial_partial_ ## name;
#include "pas_segregated_page_config_kind.def"
#undef PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND
    }
    PAS_ASSERT(!"Should not be reached");
    return (pas_local_allocator_config_kind)0;
}

static PAS_ALWAYS_INLINE pas_local_allocator_config_kind
pas_local_allocator_config_kind_create_bitfit(pas_bitfit_page_config_kind kind)
{
    switch (kind) {
#define PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND(name, value) \
    case pas_bitfit_page_config_kind_ ## name: \
        return pas_local_allocator_config_kind_bitfit_ ## name;
#include "pas_bitfit_page_config_kind.def"
#undef PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND
    }
    PAS_ASSERT(!"Should not be reached");
    return (pas_local_allocator_config_kind)0;
}

static PAS_ALWAYS_INLINE pas_segregated_page_config_kind
pas_local_allocator_config_kind_get_segregated_page_config_kind(pas_local_allocator_config_kind kind)
{
    switch (kind) {
#define PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND(name, value) \
    case pas_local_allocator_config_kind_normal_ ## name: \
    case pas_local_allocator_config_kind_primordial_partial_ ## name: \
        return pas_segregated_page_config_kind_ ## name;
#include "pas_segregated_page_config_kind.def"
#undef PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND
    default:
        PAS_ASSERT(!"Should not be reached");
        return (pas_segregated_page_config_kind)0;
    }
}

static PAS_ALWAYS_INLINE pas_bitfit_page_config_kind
pas_local_allocator_config_kind_get_bitfit_page_config_kind(pas_local_allocator_config_kind kind)
{
    switch (kind) {
#define PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND(name, value) \
    case pas_local_allocator_config_kind_bitfit_ ## name: \
        return pas_bitfit_page_config_kind_ ## name;
#include "pas_bitfit_page_config_kind.def"
#undef PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND
    default:
        PAS_ASSERT(!"Should not be reached");
        return (pas_bitfit_page_config_kind)0;
    }
}

static inline const char*
pas_local_allocator_config_kind_get_string(pas_local_allocator_config_kind kind)
{
    switch (kind) {
    case pas_local_allocator_config_kind_null:
        return "null";
    case pas_local_allocator_config_kind_unselected:
        return "unselected";
#define PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND(name, value) \
    case pas_local_allocator_config_kind_normal_ ## name: \
        return "normal_" #name; \
    case pas_local_allocator_config_kind_primordial_partial_ ## name: \
        return "primordial_partial_" #name;
#include "pas_segregated_page_config_kind.def"
#undef PAS_DEFINE_SEGREGATED_PAGE_CONFIG_KIND
#define PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND(name, value) \
    case pas_local_allocator_config_kind_bitfit_ ## name: \
        return "bitfit_" #name;
#include "pas_bitfit_page_config_kind.def"
#undef PAS_DEFINE_BITFIT_PAGE_CONFIG_KIND
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_LOCAL_ALLOCATOR_CONFIG_KIND_H */

