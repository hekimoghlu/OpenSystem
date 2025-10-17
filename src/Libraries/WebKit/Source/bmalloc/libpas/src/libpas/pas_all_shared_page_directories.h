/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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
#ifndef PAS_ALL_SHARED_PAGE_DIRECTORIES_H
#define PAS_ALL_SHARED_PAGE_DIRECTORIES_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_segregated_shared_page_directory;
typedef struct pas_segregated_shared_page_directory pas_segregated_shared_page_directory;

PAS_API extern pas_segregated_shared_page_directory* pas_first_shared_page_directory;

/* Must hold the heap lock to interact with any of this. */

PAS_API void pas_all_shared_page_directories_add(pas_segregated_shared_page_directory* directory);

typedef bool (*pas_all_shared_page_directories_callback)(
    pas_segregated_shared_page_directory* directory,
    void* arg);

PAS_API bool pas_all_shared_page_directories_for_each(
    pas_all_shared_page_directories_callback callback,
    void* arg);

PAS_END_EXTERN_C;

#endif /* PAS_ALL_SHARED_PAGE_DIRECTORIES_H */


