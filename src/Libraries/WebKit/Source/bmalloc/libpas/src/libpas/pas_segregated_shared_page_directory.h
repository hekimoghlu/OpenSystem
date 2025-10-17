/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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
#ifndef PAS_SEGREGATED_SHARED_PAGE_DIRECTORY_H
#define PAS_SEGREGATED_SHARED_PAGE_DIRECTORY_H

#include "pas_segregated_directory.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_segregated_shared_page_directory;
typedef struct pas_segregated_shared_page_directory pas_segregated_shared_page_directory;

struct pas_segregated_shared_page_directory {
    pas_segregated_directory base;
    pas_segregated_shared_page_directory* next;
    void* dump_arg;
};

#define PAS_SEGREGATED_SHARED_PAGE_DIRECTORY_INITIALIZER(page_config, sharing_mode, passed_dump_arg) \
    ((pas_segregated_shared_page_directory){ \
         .base = PAS_SEGREGATED_DIRECTORY_INITIALIZER( \
                     (page_config).kind, (sharing_mode), pas_segregated_shared_page_directory_kind), \
         .next = NULL, \
         .dump_arg = (passed_dump_arg) \
     })

extern PAS_API unsigned pas_segregated_shared_page_directory_probability_of_ineligibility;

PAS_API pas_segregated_shared_view* pas_segregated_shared_page_directory_find_first_eligible(
    pas_segregated_shared_page_directory* directory,
    unsigned size,
    unsigned alignment,
    pas_lock_hold_mode heap_lock_hold_mode);

PAS_API pas_page_sharing_pool_take_result
pas_segregated_shared_page_directory_take_last_empty(
    pas_segregated_shared_page_directory* directory,
    pas_deferred_decommit_log* log,
    pas_lock_hold_mode heap_lock_hold_mode);

PAS_API void pas_segregated_shared_page_directory_dump_reference(
    pas_segregated_shared_page_directory* directory,
    pas_stream* stream);

PAS_API void pas_segregated_shared_page_directory_dump_for_spectrum(
    pas_stream* stream, void* directory);

PAS_END_EXTERN_C;

#endif /* PAS_SEGREGATED_SHARED_PAGE_DIRECTORY_H */

