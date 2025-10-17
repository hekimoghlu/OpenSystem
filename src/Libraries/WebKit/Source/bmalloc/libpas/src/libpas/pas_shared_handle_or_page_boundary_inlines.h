/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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
#ifndef PAS_SHARED_HANDLE_OR_PAGE_BOUNDARY_INLINES_H
#define PAS_SHARED_HANDLE_OR_PAGE_BOUNDARY_INLINES_H

#include "pas_shared_handle_or_page_boundary.h"

#include "pas_segregated_page.h"
#include "pas_segregated_shared_handle.h"
#include "pas_utility_heap_config.h"

PAS_BEGIN_EXTERN_C;

static PAS_ALWAYS_INLINE pas_shared_handle_or_page_boundary
pas_wrap_shared_handle(pas_segregated_shared_handle* handle,
                       pas_segregated_page_config page_config)
{
    if (!pas_segregated_page_config_is_utility(page_config)) {
        PAS_TESTING_ASSERT(pas_segregated_page_is_allocated(
                               (uintptr_t)handle, PAS_UTILITY_HEAP_CONFIG.small_segregated_config));
    }
    return (pas_shared_handle_or_page_boundary)((uintptr_t)handle | PAS_IS_SHARED_HANDLE_BIT);
}

static inline pas_segregated_shared_handle*
pas_unwrap_shared_handle_no_liveness_checks(pas_shared_handle_or_page_boundary shared_handle_or_page)
{
    PAS_ASSERT(pas_is_wrapped_shared_handle(shared_handle_or_page));
    return (pas_segregated_shared_handle*)(
        (uintptr_t)shared_handle_or_page & ~PAS_IS_SHARED_HANDLE_BIT);
}

static PAS_ALWAYS_INLINE pas_segregated_shared_handle*
pas_unwrap_shared_handle(pas_shared_handle_or_page_boundary shared_handle_or_page,
                         pas_segregated_page_config page_config)
{
    pas_segregated_shared_handle* result;
    result = pas_unwrap_shared_handle_no_liveness_checks(shared_handle_or_page);
    if (!pas_segregated_page_config_is_utility(page_config)) {
        PAS_TESTING_ASSERT(pas_segregated_page_is_allocated(
                               (uintptr_t)result, PAS_UTILITY_HEAP_CONFIG.small_segregated_config));
    }
    return result;
}

static PAS_ALWAYS_INLINE void*
pas_shared_handle_or_page_boundary_get_page_boundary_no_liveness_checks(
    pas_shared_handle_or_page_boundary shared_handle_or_page)
{
    if (pas_is_wrapped_shared_handle(shared_handle_or_page))
        return pas_unwrap_shared_handle_no_liveness_checks(shared_handle_or_page)->page_boundary;

    return pas_unwrap_page_boundary(shared_handle_or_page);
}

static PAS_ALWAYS_INLINE void*
pas_shared_handle_or_page_boundary_get_page_boundary(
    pas_shared_handle_or_page_boundary shared_handle_or_page,
    pas_segregated_page_config page_config)
{
    if (pas_is_wrapped_shared_handle(shared_handle_or_page))
        return pas_unwrap_shared_handle(shared_handle_or_page, page_config)->page_boundary;

    return pas_unwrap_page_boundary(shared_handle_or_page);
}

PAS_END_EXTERN_C;

#endif /* PAS_SHARED_HANDLE_OR_PAGE_BOUNDARY_INLINES_H */

