/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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
#ifndef PAS_ENUMERABLE_RANGE_LIST_H
#define PAS_ENUMERABLE_RANGE_LIST_H

#include "pas_compact_atomic_enumerable_range_list_chunk_ptr.h"
#include "pas_range.h"

PAS_BEGIN_EXTERN_C;

struct pas_enumerable_range_list;
struct pas_enumerable_range_list_chunk;
typedef struct pas_enumerable_range_list pas_enumerable_range_list;
typedef struct pas_enumerable_range_list_chunk pas_enumerable_range_list_chunk;

#define PAS_ENUMERABLE_RANGE_LIST_CHUNK_SIZE 10

struct pas_enumerable_range_list {
    pas_compact_atomic_enumerable_range_list_chunk_ptr head;
};

struct pas_enumerable_range_list_chunk {
    pas_compact_atomic_enumerable_range_list_chunk_ptr next;
    unsigned num_entries;
    pas_range entries[PAS_ENUMERABLE_RANGE_LIST_CHUNK_SIZE];
};

PAS_API void pas_enumerable_range_list_append(pas_enumerable_range_list* list,
                                              pas_range range);

typedef bool (*pas_enumerable_range_list_iterate_callback)(pas_range range,
                                                           void* arg);

PAS_API bool pas_enumerable_range_list_iterate(
    pas_enumerable_range_list* list,
    pas_enumerable_range_list_iterate_callback callback,
    void* arg);

typedef bool (*pas_enumerable_range_list_iterate_remote_callback)(pas_enumerator* enumerator,
                                                                  pas_range range,
                                                                  void* arg);

PAS_API bool pas_enumerable_range_list_iterate_remote(
    pas_enumerable_range_list* remote_list,
    pas_enumerator* enumerator,
    pas_enumerable_range_list_iterate_remote_callback callback,
    void* arg);

PAS_END_EXTERN_C;

#endif /* PAS_ENUMERABLE_RANGE_LIST_H */

