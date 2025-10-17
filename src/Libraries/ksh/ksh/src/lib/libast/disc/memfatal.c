/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 2, 2023.
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
 * install error message handler for fatal malloc exceptions
 */

#include <ast.h>
#include <error.h>
#include <vmalloc.h>

#include "FEATURE/vmalloc"

#if _std_malloc

void
memfatal(void)
{
}

#else

/*
 * print message and fail on VM_BADADDR,VM_NOMEM
 */

static int
nomalloc(Vmalloc_t* region, int type, void* obj, Vmdisc_t* disc)
{
	Vmstat_t	st;

	NoP(disc);
	switch (type)
	{
#ifdef VM_BADADDR
	case VM_BADADDR:
		error(ERROR_SYSTEM|3, "invalid pointer %p passed to free or realloc", obj);
		return(-1);
#endif
	case VM_NOMEM:
		vmstat(region, &st);
		error(ERROR_SYSTEM|3, "storage allocator out of space on %lu byte request ( region %lu segments %lu busy %lu:%lu:%lu free %lu:%lu:%lu )", (size_t)obj, st.extent, st.n_seg, st.n_busy, st.s_busy, st.m_busy, st.n_free, st.s_free, st.m_free);
		return(-1);
	}
	return(0);
}

/*
 * initialize the malloc exception handler
 */

void
memfatal(void)
{
	Vmdisc_t*	disc;

	malloc(0);
	if (disc = vmdisc(Vmregion, NiL))
		disc->exceptf = nomalloc;
}

#endif
