/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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

void _STUB_vmdisc(){}

#else

#include	"vmhdr.h"

/*	Change the discipline for a region.  The old discipline
**	is returned.  If the new discipline is NULL then the
**	discipline is not changed.
**
**	Written by Kiem-Phong Vo, kpv@research.att.com, 01/16/94.
*/
#if __STD_C
Vmdisc_t* vmdisc(Vmalloc_t* vm, Vmdisc_t* disc)
#else
Vmdisc_t* vmdisc(vm, disc)
Vmalloc_t*	vm;
Vmdisc_t*	disc;
#endif
{
	Vmdisc_t*	old = vm->disc;

	if(disc)
	{	if(old->exceptf &&
		   (*old->exceptf)(vm,VM_DISC,(Void_t*)disc,old) != 0 )
			return NIL(Vmdisc_t*);
		vm->disc = disc;
	}
	return old;
}

#endif
