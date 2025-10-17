/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 12, 2024.
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
#ifndef PAS_EXCLUSIVE_VIEW_TEMPLATE_MEMO_TABLE_H
#define PAS_EXCLUSIVE_VIEW_TEMPLATE_MEMO_TABLE_H

#include "pas_config.h"

#include "pas_hashtable.h"
#include "pas_segregated_size_directory.h"

PAS_BEGIN_EXTERN_C;

struct pas_exclusive_view_template_memo_key;
typedef struct pas_exclusive_view_template_memo_key pas_exclusive_view_template_memo_key;

struct pas_exclusive_view_template_memo_key {
    unsigned object_size;
    pas_segregated_page_config_kind page_config_kind;
};

static inline pas_exclusive_view_template_memo_key
pas_exclusive_view_template_memo_key_create(
    unsigned object_size,
    pas_segregated_page_config_kind page_config_kind)
{
    pas_exclusive_view_template_memo_key result;
    result.object_size = object_size;
    result.page_config_kind = page_config_kind;
    return result;
}

typedef pas_segregated_size_directory* pas_exclusive_view_template_memo_entry;

static inline pas_exclusive_view_template_memo_entry
pas_exclusive_view_template_memo_entry_create_empty(void)
{
    return NULL;
}

static inline pas_exclusive_view_template_memo_entry
pas_exclusive_view_template_memo_entry_create_deleted(void)
{
    return (pas_exclusive_view_template_memo_entry)(uintptr_t)1;
}

static inline bool
pas_exclusive_view_template_memo_entry_is_empty_or_deleted(
    pas_exclusive_view_template_memo_entry entry)
{
    return (uintptr_t)entry <= (uintptr_t)1;
}

static inline bool
pas_exclusive_view_template_memo_entry_is_empty(
    pas_exclusive_view_template_memo_entry entry)
{
    return !entry;
}

static inline bool
pas_exclusive_view_template_memo_entry_is_deleted(
    pas_exclusive_view_template_memo_entry entry)
{
    return entry == pas_exclusive_view_template_memo_entry_create_deleted();
}

static inline pas_exclusive_view_template_memo_key
pas_exclusive_view_template_memo_entry_get_key(
    pas_exclusive_view_template_memo_entry entry)
{
    return pas_exclusive_view_template_memo_key_create(
        entry->object_size,
        entry->base.page_config_kind);
}

static inline unsigned
pas_exclusive_view_template_memo_key_get_hash(
    pas_exclusive_view_template_memo_key key)
{
    return (unsigned)((uintptr_t)key.page_config_kind ^ pas_hash32(key.object_size));
}

static inline bool
pas_exclusive_view_template_memo_key_is_equal(
    pas_exclusive_view_template_memo_key a,
    pas_exclusive_view_template_memo_key b)
{
    return a.object_size == b.object_size
        && a.page_config_kind == b.page_config_kind;
}

/* This uses the bootstrap heap because this can be used from utility. */
PAS_CREATE_HASHTABLE(pas_exclusive_view_template_memo_table,
                     pas_exclusive_view_template_memo_entry,
                     pas_exclusive_view_template_memo_key);

PAS_API extern pas_exclusive_view_template_memo_table pas_exclusive_view_template_memo_table_instance;

PAS_END_EXTERN_C;

#endif /* PAS_EXCLUSIVE_VIEW_TEMPLATE_MEMO_TABLE_H */

