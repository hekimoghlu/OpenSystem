/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 26, 2023.
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

void _STUB_vmsegment(){}

#else

#include	"vmhdr.h"

/*	Get the segment containing this address
**
**	Written by Kiem-Phong Vo, kpv@research.att.com, 02/07/95
*/

#if __STD_C
Void_t* vmsegment(Vmalloc_t* vm, Void_t* addr)
#else
Void_t* vmsegment(vm, addr)
Vmalloc_t*	vm;	/* region	*/
Void_t*		addr;	/* address	*/
#endif
{
	Seg_t		*seg;
	Vmdata_t	*vd = vm->data;

	SETLOCK(vm, 0);

	for(seg = vd->seg; seg; seg = seg->next)
		if((Vmuchar_t*)addr >= (Vmuchar_t*)seg->addr &&
		   (Vmuchar_t*)addr <  (Vmuchar_t*)seg->baddr )
			break;

	CLRLOCK(vm, 0);

	return seg ? (Void_t*)seg->addr : NIL(Void_t*);
}

#endif
