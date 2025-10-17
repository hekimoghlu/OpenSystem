/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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

#include "pas_redundant_local_allocator_node.h"

#include "pas_immortal_heap.h"

pas_redundant_local_allocator_node*
pas_redundant_local_allocator_node_create(pas_segregated_size_directory* directory)
{
    pas_redundant_local_allocator_node* result;

    result = pas_immortal_heap_allocate(
        sizeof(pas_redundant_local_allocator_node),
        "pas_redundant_local_allocator_node",
        pas_object_allocation);

    pas_compact_segregated_size_directory_ptr_store(&result->directory, directory);
    result->allocator_index = 0;

    return result;
}

#endif /* LIBPAS_ENABLED */
