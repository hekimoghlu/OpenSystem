/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 8, 2024.
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

void _STUB_vmdcheap(){}

#else

#include	"vmhdr.h"

/*	A discipline to get memory from the heap.
**
**	Written by Kiem-Phong Vo, kpv@research.att.com, 01/16/94.
*/
#if __STD_C
static Void_t* heapmem(Vmalloc_t* vm, Void_t* caddr,
			size_t csize, size_t nsize,
			Vmdisc_t* disc)
#else
static Void_t* heapmem(vm, caddr, csize, nsize, disc)
Vmalloc_t*	vm;	/* region doing allocation from 	*/
Void_t*		caddr;	/* current low address			*/
size_t		csize;	/* current size				*/
size_t		nsize;	/* new size				*/
Vmdisc_t*	disc;	/* discipline structure			*/
#endif
{
	if(csize == 0 && nsize == 0)
		return NIL(Void_t*);
	else if(csize == 0)
		return vmalloc(Vmheap,nsize);
	else if(nsize == 0)
		return vmfree(Vmheap,caddr) >= 0 ? caddr : NIL(Void_t*);
	else	return vmresize(Vmheap,caddr,nsize,0);
}

static Vmdisc_t _Vmdcheap = { heapmem, NIL(Vmexcept_f), 0 };
__DEFINE__(Vmdisc_t*,Vmdcheap,&_Vmdcheap);

#ifdef NoF
NoF(vmdcheap)
#endif

#endif
