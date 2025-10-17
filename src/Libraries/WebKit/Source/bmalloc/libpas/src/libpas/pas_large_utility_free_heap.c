/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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

#include "pas_large_utility_free_heap.h"

#include "pas_bootstrap_free_heap.h"
#include "pas_large_free_heap_helpers.h"
#include "pas_page_malloc.h"

#define PAS_LARGE_FREE_HEAP_NAME pas_large_utility_free_heap
#define PAS_LARGE_FREE_HEAP_ID(suffix) pas_large_utility_free_heap ## suffix
#define PAS_LARGE_FREE_HEAP_MEMORY_SOURCE pas_bootstrap_free_heap_try_allocate_with_alignment
#include "pas_large_free_heap_definitions.def"
#undef PAS_LARGE_FREE_HEAP_NAME
#undef PAS_LARGE_FREE_HEAP_ID

#endif /* LIBPAS_ENABLED */
