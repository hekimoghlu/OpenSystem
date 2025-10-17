/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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

#include "pas_all_shared_page_directories.h"

#include "pas_heap_lock.h"
#include "pas_segregated_shared_page_directory.h"

pas_segregated_shared_page_directory* pas_first_shared_page_directory = NULL;

void pas_all_shared_page_directories_add(pas_segregated_shared_page_directory* directory)
{
    pas_heap_lock_assert_held();
    
    PAS_ASSERT(!directory->next);
    PAS_ASSERT(pas_first_shared_page_directory != directory);
    
    directory->next = pas_first_shared_page_directory;
    pas_first_shared_page_directory = directory;
}

bool pas_all_shared_page_directories_for_each(pas_all_shared_page_directories_callback callback,
                                              void *arg)
{
    pas_segregated_shared_page_directory* directory;
    
    pas_heap_lock_assert_held();

    for (directory = pas_first_shared_page_directory; directory; directory = directory->next) {
        if (!callback(directory, arg))
            return false;
    }

    return true;
}

#endif /* LIBPAS_ENABLED */
