/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 26, 2023.
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
#ifndef _S_PTRLIST_H
#define _S_PTRLIST_H

#include <mach/boolean.h>

/* the initial number of elements in the list */
#define PTRLIST_NUMBER		16

typedef struct {
    void * *	array;	/* malloc'd array of pointers */
    int		size;	/* number of elements in array */
    int		count;	/* number of occupied elements */
} ptrlist_t;

void		ptrlist_init(ptrlist_t * list);
void		ptrlist_init_size(ptrlist_t * list, int size);
boolean_t	ptrlist_add(ptrlist_t * list, void * element);
boolean_t	ptrlist_insert(ptrlist_t * list, void * element, int i);
void		ptrlist_free(ptrlist_t * list);
boolean_t	ptrlist_dup(ptrlist_t * dest, ptrlist_t * source);
boolean_t	ptrlist_concat(ptrlist_t * list, ptrlist_t * extra);
int		ptrlist_count(ptrlist_t * list);
void *		ptrlist_element(ptrlist_t * list, int i);
boolean_t	ptrlist_remove(ptrlist_t * list, int i, void * * ret);
int		ptrlist_index(ptrlist_t * list, void * element);

#endif /* _S_PTRLIST_H */
