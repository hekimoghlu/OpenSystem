/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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

void _STUB_vmregion(){}

#else

#include	"vmhdr.h"

/*	Return the containing region of an allocated piece of memory.
**	Beware: this only works with Vmbest, Vmdebug and Vmprofile.
**
**	10/31/2009: Add handling of shared/persistent memory regions.
**
**	Written by Kiem-Phong Vo, kpv@research.att.com, 01/16/94.
*/
#if __STD_C
Vmalloc_t* vmregion(Void_t* addr)
#else
Vmalloc_t* vmregion(addr)
Void_t*	addr;
#endif
{
	Vmalloc_t	*vm;
	Vmdata_t	*vd;

	if(!addr)
		return NIL(Vmalloc_t*);

	vd = SEG(BLOCK(addr))->vmdt;

	_vmlock(NIL(Vmalloc_t*), 1);
	for(vm = Vmheap; vm; vm = vm->next)
		if(vm->data == vd)
			break;
	_vmlock(NIL(Vmalloc_t*), 0);

	return vm;
}

#endif
