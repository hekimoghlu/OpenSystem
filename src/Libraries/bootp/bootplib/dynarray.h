/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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
#ifndef _S_DYNARRAY_H
#define _S_DYNARRAY_H

/*
 * dynarray.h
 * - simple array "object" that grows when needed, and has caller-supplied
 *   functions to free/duplicate elements
 */

/* 
 * Modification History
 *
 * December 6, 1999	Dieter Siegmund (dieter@apple)
 * - initial revision
 */

#include <mach/boolean.h>
#include "ptrlist.h"

typedef void dynarray_free_func_t(void *);
typedef void * dynarray_copy_func_t(void * source);

typedef struct dynarray_s {
    ptrlist_t			list;
    dynarray_free_func_t *	free_func;
    dynarray_copy_func_t *	copy_func;
} dynarray_t;

void		dynarray_init(dynarray_t * list, 
			      dynarray_free_func_t * free_func,
			      dynarray_copy_func_t * copy_func);
boolean_t	dynarray_add(dynarray_t * list, void * element);
boolean_t	dynarray_remove(dynarray_t * list, int i, void * * element_p);
boolean_t	dynarray_insert(dynarray_t * list, void * element, int i);
void		dynarray_free(dynarray_t * list);
boolean_t	dynarray_free_element(dynarray_t * list, int i);
boolean_t	dynarray_dup(dynarray_t * source, dynarray_t * copy);
int		dynarray_count(dynarray_t * list);
void *		dynarray_element(dynarray_t * list, int i);
int		dynarray_index(dynarray_t * list, void * element);

#endif /* _S_DYNARRAY_H */
