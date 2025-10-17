/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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

void _STUB_vmclose(){}

#else

#include	"vmhdr.h"

/*	Close down a region.
**
**	Written by Kiem-Phong Vo, kpv@research.att.com, 01/16/94.
*/
#if __STD_C
int vmclose(Vmalloc_t* vm)
#else
int vmclose(vm)
Vmalloc_t*	vm;
#endif
{
	Seg_t		*seg, *vmseg, *next;
	Vmalloc_t	*v, *last;
	Vmdata_t*	vd = vm->data;
	Vmdisc_t*	disc = vm->disc;
	int		mode, rv = 0;

	if(vm == Vmheap) /* the heap is never freed */
		return -1;

	if(vm->disc->exceptf && /* announcing closing event */
	   (rv = (*vm->disc->exceptf)(vm,VM_CLOSE,(Void_t*)1,vm->disc)) < 0 )
		return -1;

	mode = vd->mode; /* remember this in case it gets destroyed below */

	if((mode&VM_MTPROFILE) && _Vmpfclose)
		(*_Vmpfclose)(vm);

	/* remove from linked list of regions */
	_vmlock(NIL(Vmalloc_t*), 1);
	for(last = Vmheap, v = last->next; v; last = v, v = v->next)
	{	if(v == vm)
		{	last->next = v->next;
			break;
		}
	}
	_vmlock(NIL(Vmalloc_t*), 0);

	if(rv == 0) /* deallocate memory obtained from the system */
	{	/* lock-free because alzheimer can cause deadlocks :) */
		vmseg = NIL(Seg_t*);
		for(seg = vd->seg; seg; seg = next)
		{	next = seg->next;
			if(seg->extent == seg->size) /* root segment */
				vmseg = seg; /* don't free this yet */
			else	(*disc->memoryf)(vm,seg->addr,seg->extent,0,disc);
		}
		if(vmseg) /* now safe to free root segment */
			(*disc->memoryf)(vm,vmseg->addr,vmseg->extent,0,disc);
	}

	if(disc->exceptf) /* finalizing closing */
		(void)(*disc->exceptf)(vm, VM_ENDCLOSE, (Void_t*)0, disc);

	if(!(mode & VM_MEMORYF) )
		vmfree(Vmheap,vm);

	return 0;
}

#endif
