/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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
#ifndef HOTBIT_HEAP_H
#define HOTBIT_HEAP_H

#include "pas_reallocate_free_mode.h"
#include "pas_allocation_mode.h"

#if PAS_ENABLE_HOTBIT

PAS_BEGIN_EXTERN_C;

PAS_API void* hotbit_try_allocate(size_t size, pas_allocation_mode allocation_mode);
PAS_API void* hotbit_try_allocate_with_alignment(size_t size,
                                                  size_t alignment,
                                                  pas_allocation_mode allocation_mode);

PAS_API void* hotbit_try_reallocate(void* old_ptr, size_t new_size,
                                     pas_reallocate_free_mode free_mode,
                                    pas_allocation_mode allocation_mode);

PAS_API void* hotbit_reallocate(void* old_ptr, size_t new_size,
                                 pas_reallocate_free_mode free_mode);

PAS_API void hotbit_deallocate(void*);

PAS_END_EXTERN_C;

#endif /* PAS_ENABLE_HOTBIT */

#endif /* HOTBIT_HEAP_H */

