/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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

#include "pas_commit_span.h"

#include "pas_deferred_decommit_log.h"
#include "pas_log.h"
#include "pas_page_base.h"
#include "pas_page_malloc.h"

static const bool verbose = false;

void pas_commit_span_construct(pas_commit_span* span, pas_mmap_capability mmap_capability)
{
    if (verbose)
        pas_log("%p: creating commit span.\n", span);
    span->index_of_start_of_span = UINTPTR_MAX;
    span->did_add_first = false;
    span->total_bytes = 0;
    span->mmap_capability = mmap_capability;
}

void pas_commit_span_add_to_change(pas_commit_span* span, uintptr_t granule_index)
{
    if (span->index_of_start_of_span == UINTPTR_MAX)
        span->index_of_start_of_span = granule_index;
    else
        PAS_ASSERT(span->index_of_start_of_span < granule_index);
}

void pas_commit_span_add_unchanged(pas_commit_span* span,
                                   pas_page_base* page,
                                   uintptr_t granule_index,
                                   const pas_page_base_config* config,
                                   void (*commit_or_decommit)(void* base, size_t size, void* arg),
                                   void* arg)
{
    size_t size;
    
    if (span->index_of_start_of_span == UINTPTR_MAX)
        return;
    
    if (verbose)
        pas_log("%p: adding a thing.\n", span);

    PAS_ASSERT(span->index_of_start_of_span < granule_index);

    size = (granule_index - span->index_of_start_of_span) * config->granule_size;
    
    commit_or_decommit(
        (char*)pas_page_base_boundary(page, *config)
        + span->index_of_start_of_span * config->granule_size,
        size,
        arg);
    span->index_of_start_of_span = UINTPTR_MAX;
    span->did_add_first = true;
    span->total_bytes += size;
}

static void commit(void* base, size_t size, void* arg)
{
    pas_commit_span* span;

    span = arg;
    
    pas_page_malloc_commit(base, size, span->mmap_capability);
}

void pas_commit_span_add_unchanged_and_commit(pas_commit_span* span,
                                              pas_page_base* page,
                                              uintptr_t granule_index,
                                              const pas_page_base_config* config)
{
    pas_commit_span_add_unchanged(span, page, granule_index, config, commit, span);
}

typedef struct {
    pas_commit_span* span;
    pas_deferred_decommit_log* log;
    pas_lock* commit_lock;
    pas_lock_hold_mode heap_lock_hold_mode;
} decommit_data;

static void decommit(void* base, size_t size, void* arg)
{
    decommit_data* data;

    data = arg;

    if (verbose)
        pas_log("did_add_first = %d, base = %p, size = %zu\n", data->span->did_add_first, base, size);

    pas_deferred_decommit_log_add_already_locked(
        data->log,
        pas_virtual_range_create(
            (uintptr_t)base,
            (uintptr_t)base + size,
            data->span->did_add_first ? NULL : data->commit_lock,
            data->span->mmap_capability),
        data->heap_lock_hold_mode);
}

void pas_commit_span_add_unchanged_and_decommit(pas_commit_span* span,
                                                pas_page_base* page,
                                                uintptr_t granule_index,
                                                pas_deferred_decommit_log* log,
                                                pas_lock* commit_lock,
                                                const pas_page_base_config* config,
                                                pas_lock_hold_mode heap_lock_hold_mode)
{
    decommit_data data;
    data.span = span;
    data.log = log;
    data.commit_lock = commit_lock;
    data.heap_lock_hold_mode = heap_lock_hold_mode;
    pas_commit_span_add_unchanged(span, page, granule_index, config, decommit, &data);
}

#endif /* LIBPAS_ENABLED */
