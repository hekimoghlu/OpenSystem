/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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
#ifndef PAS_SHARED_PAGE_DIRECTORY_BY_SIZE_H
#define PAS_SHARED_PAGE_DIRECTORY_BY_SIZE_H

#include "pas_page_sharing_mode.h"
#include "pas_segregated_shared_page_directory.h"

PAS_BEGIN_EXTERN_C;

struct pas_segregated_page_config;
struct pas_shared_page_directory_by_size;
struct pas_shared_page_directory_by_size_data;
typedef struct pas_segregated_page_config pas_segregated_page_config;
typedef struct pas_shared_page_directory_by_size pas_shared_page_directory_by_size;
typedef struct pas_shared_page_directory_by_size_data pas_shared_page_directory_by_size_data;

struct pas_shared_page_directory_by_size {
    /* These is a configuration parameter, which you may change before this gets first used. If
       you set it after data is not NULL, then nothing happens. */
    unsigned log_shift;
    pas_page_sharing_mode sharing_mode;

    pas_shared_page_directory_by_size_data* data;
};

struct pas_shared_page_directory_by_size_data {
    /* These are the actual settings that are being used. */
    unsigned log_shift;
    unsigned num_directories;

    pas_segregated_shared_page_directory directories[1];
};

#define PAS_SHARED_PAGE_DIRECTORY_BY_SIZE_INITIALIZER(passed_log_shift, passed_sharing_mode) \
    ((pas_shared_page_directory_by_size){ \
         .log_shift = (passed_log_shift), \
         .sharing_mode = (passed_sharing_mode), \
         .data = NULL \
     })

PAS_API pas_segregated_shared_page_directory* pas_shared_page_directory_by_size_get(
    pas_shared_page_directory_by_size* by_size,
    unsigned size,
    const pas_segregated_page_config* page_config);

PAS_API bool pas_shared_page_directory_by_size_for_each(
    pas_shared_page_directory_by_size* by_size,
    bool (*callback)(pas_segregated_shared_page_directory* directory,
                     void* arg),
    void* arg);

PAS_API bool pas_shared_page_directory_by_size_for_each_remote(
    pas_shared_page_directory_by_size* by_size,
    pas_enumerator* enumerator,
    bool (*callback)(pas_enumerator* enumerator,
                     pas_segregated_shared_page_directory* directory,
                     void* arg),
    void* arg);

PAS_API void pas_shared_page_directory_by_size_dump_directory_arg(
    pas_stream* stream,
    pas_segregated_shared_page_directory* directory);

PAS_END_EXTERN_C;

#endif /* PAS_SHARED_PAGE_DIRECTORY_BY_SIZE_H */

