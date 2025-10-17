/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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
#ifndef PAS_SEGREGATED_EXCLUSIVE_VIEW_H
#define PAS_SEGREGATED_EXCLUSIVE_VIEW_H

#include "pas_compact_segregated_size_directory_ptr.h"
#include "pas_lock.h"
#include "pas_segregated_deallocation_mode.h"
#include "pas_segregated_view.h"

PAS_BEGIN_EXTERN_C;

struct pas_segregated_exclusive_view;
struct pas_segregated_page;
struct pas_thread_local_cache;
typedef struct pas_segregated_exclusive_view pas_segregated_exclusive_view;
typedef struct pas_segregated_page pas_segregated_page;
typedef struct pas_thread_local_cache pas_thread_local_cache;

PAS_API extern size_t pas_segregated_exclusive_view_count;

struct pas_segregated_exclusive_view {
    void* page_boundary;
    
    pas_compact_segregated_size_directory_ptr directory;

    bool is_owned;

    unsigned index;

    /* This lock needs to be held before the heap lock is acquired. */
    pas_lock commit_lock;

    /* This lock can be acquired after the heap lock is held.
     
       This lock doubles as the page fallback lock.
    
       This lock's other purpose is to allow heap iteration to happen with the heap lock held.
       It's not obvious that we need that capability, but it seems like a useful escape hatch,
       if even just when we find that we need it in a debugging session. The alternative is to
       say that this is just the fallback lock and say that heap iteration grabs the commit
       lock. That would mean also having to make virtual page taking grab the commit lock.
    
       I don't think we have a story for the ordering between the page lock and the ownership
       lock, if they are different. */
    pas_lock ownership_lock;
};

static inline pas_segregated_view
pas_segregated_exclusive_view_as_view(pas_segregated_exclusive_view* view)
{
    return pas_segregated_view_create(view, pas_segregated_exclusive_view_kind);
}

static inline pas_segregated_view
pas_segregated_exclusive_view_as_ineligible_view(pas_segregated_exclusive_view* view)
{
    return pas_segregated_view_create(view, pas_segregated_ineligible_exclusive_view_kind);
}

static inline pas_segregated_view
pas_segregated_exclusive_view_as_view_non_null(pas_segregated_exclusive_view* view)
{
    return pas_segregated_view_create_non_null(view, pas_segregated_exclusive_view_kind);
}

static inline pas_segregated_view
pas_segregated_exclusive_view_as_ineligible_view_non_null(pas_segregated_exclusive_view* view)
{
    return pas_segregated_view_create_non_null(view, pas_segregated_ineligible_exclusive_view_kind);
}

PAS_API pas_segregated_exclusive_view*
pas_segregated_exclusive_view_create(
    pas_segregated_size_directory* directory,
    size_t index);

PAS_API void pas_segregated_exclusive_view_note_emptiness(
    pas_segregated_exclusive_view* view,
    pas_segregated_page* page);

PAS_API pas_heap_summary pas_segregated_exclusive_view_compute_summary(
    pas_segregated_exclusive_view* view);

PAS_API void pas_segregated_exclusive_view_install_full_use_counts(
    pas_segregated_exclusive_view* view);

PAS_API bool pas_segregated_exclusive_view_is_eligible(pas_segregated_exclusive_view* view);
PAS_API bool pas_segregated_exclusive_view_is_empty(pas_segregated_exclusive_view* view);

PAS_END_EXTERN_C;

#endif /* PAS_SEGREGATED_EXCLUSIVE_VIEW_H */

