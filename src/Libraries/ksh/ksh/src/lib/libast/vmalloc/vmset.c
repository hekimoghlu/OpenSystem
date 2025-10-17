/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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

void _STUB_vmset(){}

#else

#include	"vmhdr.h"


/*	Set the control flags for a region.
**
**	Written by Kiem-Phong Vo, kpv@research.att.com, 01/16/94.
*/
#if __STD_C
int vmset(reg Vmalloc_t* vm, int flags, int on)
#else
int vmset(vm, flags, on)
reg Vmalloc_t*	vm;	/* region being worked on		*/
int		flags;	/* flags must be in VM_FLAGS		*/
int		on;	/* !=0 if turning on, else turning off	*/
#endif
{
	int		mode;
	Vmdata_t	*vd = vm->data;

	if(flags == 0 && on == 0)
		return vd->mode;

	SETLOCK(vm, 0);

	mode = vd->mode;
	if(on)
		vd->mode |=  (flags&VM_FLAGS);
	else	vd->mode &= ~(flags&VM_FLAGS);

	CLRLOCK(vm, 0);

	return mode;
}

#endif
