/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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

#include "pas_heap_for_config.h"

#include "pas_compact_bootstrap_free_heap.h"
#include "pas_utility_heap.h"
#include "pas_utility_heap_config.h"

bool pas_heap_for_config_force_bootstrap = false;

void* pas_heap_for_config_allocate(
    const pas_heap_config* config,
    size_t size,
    const char* name)
{
    /* As a matter of policy, we say that !config means use bootstrap heap. We use that to indicate
       that we're so low in the stack that we can't do utility. But, theoretically, we ought to be
       able to make those allocation use utility if we really needed them to. */
    if (pas_heap_for_config_force_bootstrap || !config || pas_heap_config_is_utility(config))
        return pas_compact_bootstrap_free_heap_allocate_simple(size, name, pas_object_allocation);

    return pas_utility_heap_allocate(size, name);
}

void* pas_heap_for_page_config_kind_allocate(
    pas_segregated_page_config_kind page_config_kind,
    size_t size,
    const char* name)
{
    if (pas_heap_for_config_force_bootstrap
        || page_config_kind == pas_segregated_page_config_kind_null
        || pas_segregated_page_config_kind_is_utility(page_config_kind))
        return pas_compact_bootstrap_free_heap_allocate_simple(size, name, pas_object_allocation);

    return pas_utility_heap_allocate(size, name);
}

void* pas_heap_for_page_config_allocate(
    const pas_segregated_page_config* page_config,
    size_t size,
    const char* name)
{
    return pas_heap_for_page_config_kind_allocate(
        pas_segregated_page_config_get_kind(page_config), size, name);
}

void* pas_heap_for_config_allocate_with_alignment(
    const pas_heap_config* config,
    size_t size,
    size_t alignment,
    const char* name)
{
    if (pas_heap_for_config_force_bootstrap || !config || pas_heap_config_is_utility(config)) {
        return (void*)pas_compact_bootstrap_free_heap_allocate_with_alignment(
            size, pas_alignment_create_traditional(alignment), name, pas_object_allocation).begin;
    }

    return pas_utility_heap_allocate_with_alignment(size, alignment, name);
}

void* pas_heap_for_page_config_allocate_with_alignment(
    const pas_segregated_page_config* page_config,
    size_t size,
    size_t alignment,
    const char* name)
{
    if (pas_heap_for_config_force_bootstrap || !page_config
        || pas_segregated_page_config_is_utility(*page_config)) {
        return (void*)pas_compact_bootstrap_free_heap_allocate_with_alignment(
            size, pas_alignment_create_traditional(alignment), name, pas_object_allocation).begin;
    }

    return pas_utility_heap_allocate_with_alignment(size, alignment, name);
}

void* pas_heap_for_config_allocate_with_manual_alignment(
    const pas_heap_config* config,
    size_t size,
    size_t alignment,
    const char* name)
{
    if (pas_heap_for_config_force_bootstrap || !config || pas_heap_config_is_utility(config)) {
        return (void*)pas_compact_bootstrap_free_heap_allocate_with_manual_alignment(
            size, pas_alignment_create_traditional(alignment), name, pas_object_allocation).begin;
    }

    return pas_utility_heap_allocate_with_alignment(size, alignment, name);
}

void* pas_heap_for_page_config_kind_allocate_with_manual_alignment(
    pas_segregated_page_config_kind page_config_kind,
    size_t size,
    size_t alignment,
    const char* name)
{
    if (pas_heap_for_config_force_bootstrap
        || page_config_kind == pas_segregated_page_config_kind_null
        || pas_segregated_page_config_kind_is_utility(page_config_kind)) {
        return (void*)pas_compact_bootstrap_free_heap_allocate_with_manual_alignment(
            size, pas_alignment_create_traditional(alignment), name, pas_object_allocation).begin;
    }

    return pas_utility_heap_allocate_with_alignment(size, alignment, name);
}

void* pas_heap_for_page_config_allocate_with_manual_alignment(
    const pas_segregated_page_config* page_config,
    size_t size,
    size_t alignment,
    const char* name)
{
    return pas_heap_for_page_config_kind_allocate_with_manual_alignment(
        pas_segregated_page_config_get_kind(page_config), size, alignment, name);
}

void pas_heap_for_config_deallocate(
    const pas_heap_config* config,
    void* ptr,
    size_t size)
{
    if (pas_heap_for_config_force_bootstrap || !config || pas_heap_config_is_utility(config)) {
        pas_compact_bootstrap_free_heap_deallocate(ptr, size, pas_object_allocation);
        return;
    }

    pas_utility_heap_deallocate(ptr);
}

void pas_heap_for_page_config_kind_deallocate(
    pas_segregated_page_config_kind page_config_kind,
    void* ptr,
    size_t size)
{
    if (pas_heap_for_config_force_bootstrap
        || page_config_kind == pas_segregated_page_config_kind_null
        || pas_segregated_page_config_kind_is_utility(page_config_kind)) {
        pas_compact_bootstrap_free_heap_deallocate(ptr, size, pas_object_allocation);
        return;
    }

    pas_utility_heap_deallocate(ptr);
}

void pas_heap_for_page_config_deallocate(
    const pas_segregated_page_config* page_config,
    void* ptr,
    size_t size)
{
    pas_heap_for_page_config_kind_deallocate(
        pas_segregated_page_config_get_kind(page_config), ptr, size);
}

#endif /* LIBPAS_ENABLED */
