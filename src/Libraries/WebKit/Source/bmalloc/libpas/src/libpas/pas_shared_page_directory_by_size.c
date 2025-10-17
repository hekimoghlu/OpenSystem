/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 22, 2025.
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

#include "pas_shared_page_directory_by_size.h"

#include <math.h>
#include "pas_heap_lock.h"
#include "pas_immortal_heap.h"
#include "pas_segregated_page_config.h"
#include "pas_stream.h"

pas_segregated_shared_page_directory* pas_shared_page_directory_by_size_get(
    pas_shared_page_directory_by_size* by_size,
    unsigned size,
    const pas_segregated_page_config* page_config)
{
    pas_shared_page_directory_by_size_data* data;
    unsigned index;

    data = by_size->data;
    if (PAS_UNLIKELY(!data)) {
        unsigned log_shift;
        unsigned min_size;
        unsigned max_size;
        unsigned num_directories;
        unsigned max_index;

        log_shift = by_size->log_shift;
        
        min_size = (unsigned)pas_segregated_page_config_min_align(*page_config);
        max_size = (unsigned)page_config->base.max_object_size;

        PAS_ASSERT(size >= min_size);
        PAS_ASSERT(size <= max_size);

        max_index = pas_log2_rounded_up_safe(
            max_size >> page_config->base.min_align_shift) >> log_shift;
        
        PAS_ASSERT(max_index <= max_size - min_size);

        num_directories = (unsigned)max_index + 1;

        pas_heap_lock_lock();

        data = by_size->data;
        if (data) {
            PAS_ASSERT(data->log_shift == log_shift);
            PAS_ASSERT(data->num_directories == num_directories);
        } else {
            unsigned index;
            
            data = pas_immortal_heap_allocate(
                PAS_OFFSETOF(pas_shared_page_directory_by_size_data, directories)
                + sizeof(pas_segregated_shared_page_directory) * num_directories,
                "pas_shared_page_directory_by_size_data",
                pas_object_allocation);

            data->log_shift = log_shift;
            data->num_directories = num_directories;

            for (index = num_directories; index--;) {
                data->directories[index] = PAS_SEGREGATED_SHARED_PAGE_DIRECTORY_INITIALIZER(
                    *page_config, by_size->sharing_mode,
                    (void*)(((size_t)1 << ((size_t)index << log_shift)) << page_config->base.min_align_shift));
            }

            pas_fence();

            by_size->data = data;
        }
        
        pas_heap_lock_unlock();
    }

    index = pas_log2_rounded_up_safe(size >> page_config->base.min_align_shift) >> data->log_shift;

    PAS_ASSERT(index < data->num_directories);

    return data->directories + index;
}

bool pas_shared_page_directory_by_size_for_each(
    pas_shared_page_directory_by_size* by_size,
    bool (*callback)(pas_segregated_shared_page_directory* directory,
                     void* arg),
    void* arg)
{
    pas_shared_page_directory_by_size_data* data;
    unsigned index;

    data = by_size->data;
    if (!data)
        return true;

    for (index = data->num_directories; index--;) {
        if (!callback(data->directories + index, arg))
            return false;
    }

    return true;
}

bool pas_shared_page_directory_by_size_for_each_remote(
    pas_shared_page_directory_by_size* by_size,
    pas_enumerator* enumerator,
    bool (*callback)(pas_enumerator* enumerator,
                     pas_segregated_shared_page_directory* directory,
                     void* arg),
    void* arg)
{
    pas_shared_page_directory_by_size_data* data;
    unsigned index;

    data = pas_enumerator_read_compact(enumerator, by_size->data);
    if (!data)
        return true;

    for (index = data->num_directories; index--;) {
        if (!callback(enumerator, data->directories + index, arg))
            return false;
    }

    return true;
}

void pas_shared_page_directory_by_size_dump_directory_arg(
    pas_stream* stream,
    pas_segregated_shared_page_directory* directory)
{
    pas_stream_printf(stream, "Size = %zu", (size_t)directory->dump_arg);
}

#endif /* LIBPAS_ENABLED */
