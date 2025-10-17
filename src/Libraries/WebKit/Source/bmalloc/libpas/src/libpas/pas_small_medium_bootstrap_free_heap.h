/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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
#ifndef PAS_SMALL_MEDIUM_BOOTSTRAP_FREE_HEAP_H
#define PAS_SMALL_MEDIUM_BOOTSTRAP_FREE_HEAP_H

#include "pas_allocation_config.h"
#include "pas_allocation_kind.h"
#include "pas_lock.h"
#include "pas_simple_large_free_heap.h"

PAS_BEGIN_EXTERN_C;

#define PAS_BOOTSTRAP_FOR_SMALL_FREE_LIST_MINIMUM_SIZE 4u

#define PAS_SIMPLE_FREE_HEAP_NAME pas_small_medium_bootstrap_free_heap
#define PAS_SIMPLE_FREE_HEAP_ID(suffix) pas_small_medium_bootstrap_free_heap ## suffix
#include "pas_simple_free_heap_declarations.def"
#undef PAS_SIMPLE_FREE_HEAP_NAME
#undef PAS_SIMPLE_FREE_HEAP_ID

PAS_END_EXTERN_C;

#endif /* PAS_SMALL_MEDIUM_BOOTSTRAP_FREE_HEAP_H */
