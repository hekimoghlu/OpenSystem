/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 25, 2024.
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
/*
 * an abstract heap implementation
 */

#ifndef _HEIM_HEAP_H
#define _HEIM_HEAP_H 1

typedef int (*heap_cmp_fn)(const void *, const void *);

typedef unsigned heap_ptr;

#define HEAP_INVALID_PTR ((heap_ptr)-1)

struct heap_element {
    const void *data;
    heap_ptr *ptr;
};

typedef struct heap_element heap_element;

typedef struct heap Heap;

Heap *heap_new (unsigned sz, heap_cmp_fn cmp);

int
heap_insert (Heap *h, const void *data, heap_ptr *ptr);

const void *
heap_head (Heap *h);

void
heap_remove_head (Heap *h);

int
heap_remove (Heap *h, heap_ptr ptr);

void
heap_delete (Heap *h);

int
heap_verify (Heap *h);

#endif /* _HEIM_HEAP_H */
