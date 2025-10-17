/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#if defined(_UWIN) && defined(_BLD_ast)

void _STUB_vmexit(){}

#else

#include	"vmhdr.h"

/*
**	Any required functions for process exiting.
**	Written by Kiem-Phong Vo, kpv@research.att.com (05/25/93).
*/
#if _PACKAGE_ast || _lib_atexit

void _STUB_vmexit(){}

#else

#if _lib_onexit

#if __STD_C
int atexit(void (*exitf)(void))
#else
int atexit(exitf)
void	(*exitf)();
#endif
{
	return onexit(exitf);
}

#else /*!_lib_onexit*/

typedef struct _exit_s
{	struct _exit_s*	next;
	void(*		exitf)_ARG_((void));
} Exit_t;
static Exit_t*	Exit;

#if __STD_C
atexit(void (*exitf)(void))
#else
atexit(exitf)
void	(*exitf)();
#endif
{	Exit_t*	e;

	if(!(e = (Exit_t*)malloc(sizeof(Exit_t))) )
		return -1;
	e->exitf = exitf;
	e->next = Exit;
	Exit = e;
	return 0;
}

#if __STD_C
void exit(int type)
#else
void exit(type)
int	type;
#endif
{
	Exit_t*	e;

	for(e = Exit; e; e = e->next)
		(*e->exitf)();

#if _exit_cleanup
	_cleanup();
#endif

	_exit(type);
	return type;
}

#endif	/* _lib_onexit || _lib_on_exit */

#endif /*!PACKAGE_ast*/

#endif
