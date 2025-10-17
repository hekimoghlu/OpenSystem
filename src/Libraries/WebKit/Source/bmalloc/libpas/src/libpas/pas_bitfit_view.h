/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
#ifndef PAS_BITFIT_VIEW_H
#define PAS_BITFIT_VIEW_H

#include "pas_bitfit_page_config.h"
#include "pas_compact_bitfit_directory_ptr.h"
#include "pas_heap_summary.h"
#include "pas_lock.h"

PAS_BEGIN_EXTERN_C;

struct pas_bitfit_directory;
struct pas_bitfit_page;
struct pas_bitfit_view;
typedef struct pas_bitfit_directory pas_bitfit_directory;
typedef struct pas_bitfit_page pas_bitfit_page;
typedef struct pas_bitfit_view pas_bitfit_view;

struct pas_bitfit_view {
    void* page_boundary;
    pas_compact_bitfit_directory_ptr directory;
    bool is_owned;
    unsigned index;
    pas_lock ownership_lock;
    pas_lock commit_lock;
};

PAS_API pas_bitfit_view* pas_bitfit_view_create(pas_bitfit_directory* directory,
                                                unsigned index);

PAS_API void pas_bitfit_view_note_nonemptiness(pas_bitfit_view* view);
PAS_API void pas_bitfit_view_note_full_emptiness(pas_bitfit_view* view, pas_bitfit_page* page);
PAS_API void pas_bitfit_view_note_partial_emptiness(pas_bitfit_view* view, pas_bitfit_page* page);
PAS_API void pas_bitfit_view_note_max_free(pas_bitfit_view* view);

PAS_API pas_heap_summary pas_bitfit_view_compute_summary(pas_bitfit_view* view);

typedef bool (*pas_bitfit_view_for_each_live_object_callback)(
    pas_bitfit_view* view,
    uintptr_t begin,
    size_t size,
    void* arg);

PAS_API bool pas_bitfit_view_for_each_live_object(
    pas_bitfit_view* view,
    pas_bitfit_view_for_each_live_object_callback callback,
    void* arg);

PAS_END_EXTERN_C;

#endif /* PAS_BITFIT_VIEW_H */

