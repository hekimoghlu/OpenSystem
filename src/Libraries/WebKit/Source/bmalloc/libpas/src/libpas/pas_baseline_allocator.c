/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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

#include "pas_baseline_allocator.h"

#include "pas_segregated_size_directory.h"

void pas_baseline_allocator_attach_directory(pas_baseline_allocator* allocator,
                                             pas_segregated_size_directory* directory)
{
    PAS_ASSERT(!allocator->u.allocator.view);
    
    PAS_ASSERT(
        PAS_BASELINE_LOCAL_ALLOCATOR_SIZE
        >= pas_segregated_size_directory_local_allocator_size(directory));
    
    pas_local_allocator_construct(&allocator->u.allocator, directory);
}

void pas_baseline_allocator_detach_directory(pas_baseline_allocator* allocator)
{
    PAS_ASSERT(allocator->u.allocator.view);
    pas_local_allocator_stop(
        &allocator->u.allocator,
        pas_lock_lock_mode_lock,
        pas_lock_is_not_held);
    pas_zero_memory(&allocator->u.allocator, sizeof(pas_local_allocator)); /* Does not zero the bits,
                                                                              which is OK. */
}

#endif /* LIBPAS_ENABLED */
