/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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

void _STUB_vmwalk(){}

#else

#include	"vmhdr.h"

/*	Walks all segments created in region(s)
**
**	Written by Kiem-Phong Vo, kpv@research.att.com (02/08/96)
*/

#if __STD_C
int vmwalk(Vmalloc_t* vm, int(*segf)(Vmalloc_t*, Void_t*, size_t, Vmdisc_t*, Void_t*), Void_t* handle )
#else
int vmwalk(vm, segf, handle)
Vmalloc_t*	vm;
int(*		segf)(/* Vmalloc_t*, Void_t*, size_t, Vmdisc_t*, Void_t* */);
Void_t*		handle;
#endif
{	
	reg Seg_t	*seg;
	reg int		rv = 0;

	if(!vm)
	{	_vmlock(NIL(Vmalloc_t*), 1);
		for(vm = Vmheap; vm; vm = vm->next)
		{	SETLOCK(vm, 0);
			for(seg = vm->data->seg; seg; seg = seg->next)
				if((rv = (*segf)(vm, seg->addr, seg->extent, vm->disc, handle)) < 0 )
					break;
			CLRLOCK(vm, 0);
		}
		_vmlock(NIL(Vmalloc_t*), 0);
	}
	else
	{	SETLOCK(vm, 0);
		for(seg = vm->data->seg; seg; seg = seg->next)
			if((rv = (*segf)(vm, seg->addr, seg->extent, vm->disc, handle)) < 0 )
				break;
		CLRLOCK(vm, 0);
	}

	return rv;
}

#endif
