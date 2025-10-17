/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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
 * Copyright (c) 2001 by Sun Microsystems, Inc.
 * All rights reserved.
 */

/*
 * Routines for manipulating stacks
 */

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include "stack.h"
#include "memory.h"

#define	STACK_SEEDSIZE	5

struct stk {
	int st_nument;
	int st_top;
	void **st_data;

	void (*st_free)(void *);
};

stk_t *
stack_new(void (*freep)(void *))
{
	stk_t *sp;

	sp = xmalloc(sizeof (stk_t));
	sp->st_nument = STACK_SEEDSIZE;
	sp->st_top = -1;
	sp->st_data = xmalloc(sizeof (void *) * sp->st_nument);
	sp->st_free = freep;

	return (sp);
}

void
stack_free(stk_t *sp)
{
	int i;

	if (sp->st_free) {
		for (i = 0; i <= sp->st_top; i++)
			sp->st_free(sp->st_data[i]);
	}
	free(sp->st_data);
	free(sp);
}

void *
stack_pop(stk_t *sp)
{
	assert(sp->st_top >= 0);

	return (sp->st_data[sp->st_top--]);
}

void *
stack_peek(stk_t *sp)
{
	if (sp->st_top == -1)
		return (NULL);

	return (sp->st_data[sp->st_top]);
}

void
stack_push(stk_t *sp, void *data)
{
	sp->st_top++;

	if (sp->st_top == sp->st_nument) {
		sp->st_nument += STACK_SEEDSIZE;
		sp->st_data = xrealloc(sp->st_data,
		    sizeof (void *) * sp->st_nument);
	}

	sp->st_data[sp->st_top] = data;
}

int
stack_level(stk_t *sp)
{
	return (sp->st_top + 1);
}
