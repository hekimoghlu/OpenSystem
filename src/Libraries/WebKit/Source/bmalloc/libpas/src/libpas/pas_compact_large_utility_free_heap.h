/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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
#ifndef PAS_COMPACT_LARGE_UTILITY_FREE_HEAP_H
#define PAS_COMPACT_LARGE_UTILITY_FREE_HEAP_H

#include "pas_allocation_kind.h"
#include "pas_fast_large_free_heap.h"
#include "pas_heap_summary.h"

PAS_BEGIN_EXTERN_C;

#define PAS_LARGE_FREE_HEAP_NAME pas_compact_large_utility_free_heap
#define PAS_LARGE_FREE_HEAP_ID(suffix) pas_compact_large_utility_free_heap ## suffix
#include "pas_large_free_heap_declarations.def"
#undef PAS_LARGE_FREE_HEAP_NAME
#undef PAS_LARGE_FREE_HEAP_ID

PAS_END_EXTERN_C;

#endif /* PAS_COMPACT_LARGE_UTILITY_FREE_HEAP_H */

