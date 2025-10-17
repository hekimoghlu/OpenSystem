/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 11, 2024.
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
__PAS_BEGIN_EXTERN_C;

struct __pas_heap_ref;
struct __pas_heap_type;
struct __pas_heap;
typedef struct __pas_heap_ref __pas_heap_ref;
typedef struct __pas_heap_type __pas_heap_type;
typedef struct __pas_heap __pas_heap;

struct __pas_heap_ref {
    const __pas_heap_type* type;
    __pas_heap* heap; /* initialize to NULL */
    unsigned allocator_index; /* initialize to 0 */
};

__PAS_END_EXTERN_C;

