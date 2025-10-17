/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
 * dtopen() with handle placed in specific vm region
 */

#include <dt.h>

typedef struct Dc_s
{
	Dtdisc_t	ndisc;
	Dtdisc_t*	odisc;
	Vmalloc_t*	vm;
} Dc_t;

static int
eventf(Dt_t* dt, int op, void* data, Dtdisc_t* disc)
{
	Dc_t*	dc = (Dc_t*)disc;
	int	r;

	if (dc->odisc->eventf && (r = (*dc->odisc->eventf)(dt, op, data, dc->odisc)))
		return r;
	return op == DT_ENDOPEN ? 1 : 0;
}

static void*
memoryf(Dt_t* dt, void* addr, size_t size, Dtdisc_t* disc)
{
	return vmresize(((Dc_t*)disc)->vm, addr, size, VM_RSMOVE);
}

/*
 * open a dictionary using disc->memoryf if set or vm otherwise
 */

Dt_t*
_dtnew(Vmalloc_t* vm, Dtdisc_t* disc, Dtmethod_t* meth, unsigned long version)
{
	Dt_t*		dt;
	Dc_t		dc;

	dc.odisc = disc;
	dc.ndisc = *disc;
	dc.ndisc.eventf = eventf;
	if (!dc.ndisc.memoryf)
		dc.ndisc.memoryf = memoryf;
	dc.vm = vm;
	if (dt = _dtopen(&dc.ndisc, meth, version))
		dtdisc(dt, disc, DT_SAMECMP|DT_SAMEHASH);
	return dt;
}

#undef dtnew

Dt_t*
dtnew(Vmalloc_t* vm, Dtdisc_t* disc, Dtmethod_t* meth)
{
	return _dtnew(vm, disc, meth, 20050420L);
}
