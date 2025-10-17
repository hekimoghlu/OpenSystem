/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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
 * ANSI C atexit()
 * arrange for func to be called LIFO on exit()
 */

#include <ast.h>

#if _lib_atexit

NoN(atexit)

#else

#if _lib_onexit || _lib_on_exit

#if !_lib_onexit
#define onexit		on_exit
#endif

extern int		onexit(void(*)(void));

int
atexit(void (*func)(void))
{
	return(onexit(func));
}

#else

struct list
{
	struct list*	next;
	void		(*func)(void);
};

static struct list*	funclist;

extern void		_exit(int);

int
atexit(void (*func)(void))
{
	register struct list*	p;

	if (!(p = newof(0, struct list, 1, 0))) return(-1);
	p->func = func;
	p->next = funclist;
	funclist = p;
	return(0);
}

void
_ast_atexit(void)
{
	register struct list*	p;

	while (p = funclist)
	{
		funclist = p->next;
		(*p->func)();
	}
}

#if _std_cleanup

#if _lib__cleanup
extern void		_cleanup(void);
#endif

void
exit(int code)
{
	_ast_atexit();
#if _lib__cleanup
	_cleanup();
#endif
	_exit(code);
}

#else

void
_cleanup(void)
{
	_ast_atexit();
}

#endif

#endif

#endif
