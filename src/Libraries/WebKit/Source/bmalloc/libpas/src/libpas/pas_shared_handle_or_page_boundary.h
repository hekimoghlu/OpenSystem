/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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
#ifndef PAS_SHARED_HANDLE_OR_PAGE_BOUNDARY_H
#define PAS_SHARED_HANDLE_OR_PAGE_BOUNDARY_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_segregated_page;
struct pas_segregated_shared_handle;
struct pas_shared_handle_or_page_boundary_opaque;
typedef struct pas_segregated_page pas_segregated_page;
typedef struct pas_segregated_shared_handle pas_segregated_shared_handle;
typedef struct pas_shared_handle_or_page_boundary_opaque* pas_shared_handle_or_page_boundary;

#define PAS_IS_SHARED_HANDLE_BIT ((uintptr_t)1)

static inline pas_shared_handle_or_page_boundary
pas_wrap_page_boundary(void* page_boundary)
{
    return (pas_shared_handle_or_page_boundary)page_boundary;
}

static inline bool
pas_is_wrapped_shared_handle(pas_shared_handle_or_page_boundary shared_handle_or_page)
{
    return (uintptr_t)shared_handle_or_page & PAS_IS_SHARED_HANDLE_BIT;
}

static inline bool
pas_is_wrapped_page_boundary(pas_shared_handle_or_page_boundary shared_handle_or_page)
{
    return !pas_is_wrapped_shared_handle(shared_handle_or_page);
}

static inline void*
pas_unwrap_page_boundary(pas_shared_handle_or_page_boundary shared_handle_or_page)
{
    PAS_ASSERT(pas_is_wrapped_page_boundary(shared_handle_or_page));
    return shared_handle_or_page;
}

PAS_END_EXTERN_C;

#endif /* PAS_SHARED_HANDLE_OR_PAGE_BOUNDARY_H */

