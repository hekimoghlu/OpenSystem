/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 28, 2025.
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
#ifndef PAS_SEGREGATED_SHARED_VIEW_INLINES_H
#define PAS_SEGREGATED_SHARED_VIEW_INLINES_H

#include "pas_segregated_shared_view.h"
#include "pas_shared_handle_or_page_boundary_inlines.h"

PAS_BEGIN_EXTERN_C;

/* NOTE: Right now this is responsible for actually doing the page allocation. However, it's not
   obvious that this is the right approach. You could imagine having the shared page directory do
   that. But, that would not reduce the amount of checks that commit_page_if_necessary does. */
PAS_API pas_segregated_shared_handle* pas_segregated_shared_view_commit_page(
    pas_segregated_shared_view* view,
    pas_segregated_heap* heap,
    pas_segregated_shared_page_directory* directory,
    pas_segregated_partial_view* partial_view,
    const pas_segregated_page_config* page_config);

/* Must be called holding the commit lock. */
static PAS_ALWAYS_INLINE pas_segregated_shared_handle*
pas_segregated_shared_view_commit_page_if_necessary(
    pas_segregated_shared_view* view,
    pas_segregated_heap* heap,
    pas_segregated_shared_page_directory* directory,
    pas_segregated_partial_view* partial_view,
    pas_segregated_page_config page_config)
{
    pas_shared_handle_or_page_boundary shared_handle_or_page_boundary;
    pas_segregated_shared_handle* result;

    shared_handle_or_page_boundary = view->shared_handle_or_page_boundary;
    
    /* This also checks that we've allocated a page, since we never wrap NULL as a shared handle. */
    if (pas_is_wrapped_shared_handle(shared_handle_or_page_boundary))
        result = pas_unwrap_shared_handle(shared_handle_or_page_boundary, page_config);
    else {
        result = pas_segregated_shared_view_commit_page(
            view, heap, directory, partial_view,
            (const pas_segregated_page_config*)page_config.base.page_config_ptr);
    }

    PAS_TESTING_ASSERT(result->directory == directory);

    return result;
}

PAS_END_EXTERN_C;

#endif /* PAS_SEGREGATED_SHARED_VIEW_INLINES_H */

