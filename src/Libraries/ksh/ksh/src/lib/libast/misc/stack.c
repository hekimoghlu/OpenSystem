/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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
#pragma prototyped
/*
 * pointer stack routines
 */

static const char id_stack[] = "\n@(#)$Id: stack (AT&T Bell Laboratories) 1984-05-01 $\0\n";

#include <ast.h>
#include <stack.h>

/*
 * create a new stack
 */

STACK
stackalloc(register int size, void* error)
{
	register STACK			stack;
	register struct stackblock	*b;

	if (size <= 0) size = 100;
	if (!(stack = newof(0, struct stacktable, 1, 0))) return(0);
	if (!(b = newof(0, struct stackblock, 1, 0)))
	{
		free(stack);
		return(0);
	}
	if (!(b->stack = newof(0, void*, size, 0)))
	{
		free(b);
		free(stack);
		return(0);
	}
	stack->blocks = b;
	stack->size = size;
	stack->error = error;
	stack->position.block = b;
	stack->position.index = -1;
	b->next = 0;
	b->prev = 0;
	return(stack);
}

/*
 * remove a stack
 */

void
stackfree(register STACK stack)
{
	register struct stackblock*	b;
	register struct stackblock*	p;

	b = stack->blocks;
	while (p = b)
	{
		b = p->next;
		free(p->stack);
		free(p);
	}
	free(stack);
}

/*
 * clear stack
 */

void
stackclear(register STACK stack)
{
	stack->position.block = stack->blocks;
	stack->position.index = -1;
}

/*
 * get value on top of stack
 */

void*
stackget(register STACK stack)
{
	if (stack->position.index < 0) return(stack->error);
	else return(stack->position.block->stack[stack->position.index]);
}

/*
 * push value on to stack
 */

int
stackpush(register STACK stack, void* value)
{
	register struct stackblock	*b;

	if (++stack->position.index >= stack->size)
	{
		b = stack->position.block;
		if (b->next) b = b->next;
		else
		{
			if (!(b->next = newof(0, struct stackblock, 1, 0)))
				return(-1);
			b = b->next;
			if (!(b->stack = newof(0, void*, stack->size, 0)))
				return(-1);
			b->prev = stack->position.block;
			b->next = 0;
		}
		stack->position.block = b;
		stack->position.index = 0;
	}
	stack->position.block->stack[stack->position.index] = value;
	return(0);
}

/*
 * pop value off stack
 */

int
stackpop(register STACK stack)
{
	/*
	 * return:
	 *
	 *	-1	if stack empty before pop
	 *	 0	if stack empty after pop
	 *	 1	if stack not empty before & after pop
	 */

	if (stack->position.index < 0) return(-1);
	else if (--stack->position.index < 0)
	{
		if (!stack->position.block->prev) return(0);
		stack->position.block = stack->position.block->prev;
		stack->position.index = stack->size - 1;
		return(1);
	}
	else return(1);
}

/*
 * set|get stack position
 */

void
stacktell(register STACK stack, int set, STACKPOS* position)
{
	if (set) stack->position = *position;
	else *position = stack->position;
}
