/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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
 * standalone mini vmalloc implementation
 * no resize, no free, no disciplines, no methods
 */

#include <ast.h>
#include <vmalloc.h>

Vmalloc_t*	Vmregion;

Vmalloc_t*
_vm_open(void)
{
	Vmalloc_t*	vp;

	if (vp = newof(0, Vmalloc_t, 1, 0))
	{
		vp->current = &vp->base;
		vp->data = vp->current->data;
		vp->size = sizeof(vp->current->data);
	}
	return vp;
}

int
_vm_close(register Vmalloc_t* vp)
{
	register Vmchunk_t*	cp;
	register Vmchunk_t*	np;

	if (!vp)
		return -1;
	np = vp->base.next;
	while (cp = np)
	{
		np = cp->next;
		free(cp);
	}
	free(vp);
	return 0;
}

void*
_vm_resize(register Vmalloc_t* vp, void* o, unsigned long size)
{
	char*		p;
	unsigned long	n;
	unsigned long	z;

	z = vp->last;
	vp->last = size;
	if (o && size < z)
		return o;
	if ((o ? (size - z) : size) > vp->size)
	{
		n = (size > sizeof(vp->current->data)) ? (size - sizeof(vp->current->data)) : 0;
		if (!(vp->current->next = newof(0, Vmchunk_t, 1, n)))
			return 0;
		vp->current = vp->current->next;
		vp->data = vp->current->data;
		vp->size = n ? 0 : sizeof(vp->current->data);
		if (o)
		{
			memcpy(vp->data, o, z);
			o = (void*)vp->data;
		}
	}
	else if (o)
		size -= z;
	p = vp->data;
	size = roundof(size, VM_ALIGN);
	if (size >= vp->size)
		vp->size = 0;
	else
	{
		vp->size -= size;
		vp->data += size;
	}
	return p;
}
