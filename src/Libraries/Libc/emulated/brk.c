/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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
/*
 *	File:	brk.c
 *
 *	Unix compatibility for sbrk system call.
 *
 * HISTORY
 * 09-Mar-90  Gregg Kellogg (gk) at NeXT.
 *	include <kern/mach_interface.h> instead of <kern/mach.h>
 *
 * 14-Feb-89  Avadis Tevanian (avie) at NeXT.
 *	Total rewrite using a fixed area of VM from break region.
 */

#include <mach/mach.h>		/* for vm_allocate, vm_offset_t */
#include <mach/vm_statistics.h>
#include <errno.h>
#include <unistd.h>

static int sbrk_needs_init = TRUE;
static vm_size_t sbrk_region_size = 4*1024*1024; /* Well, what should it be? */
static vm_address_t sbrk_curbrk;

void *sbrk(int size)
{
	kern_return_t	ret;

	if (sbrk_needs_init) {
		sbrk_needs_init = FALSE;
		/*
		 *	Allocate a big region to simulate break region.
		 */
		ret =  vm_allocate(mach_task_self(), &sbrk_curbrk, sbrk_region_size,
				  VM_MAKE_TAG(VM_MEMORY_SBRK)|TRUE);
		if (ret != KERN_SUCCESS) {
			errno = ENOMEM;
			return((void *)-1);
		}
	}
	
	if (size <= 0)
		return((void *)sbrk_curbrk);
	else if (size > sbrk_region_size) {
		errno = ENOMEM;
		return((void *)-1);
	}
	sbrk_curbrk += size;
	sbrk_region_size -= size;
	return((void *)(sbrk_curbrk - size));
}

void *brk(const void *x)
{
	errno = ENOMEM;
	return((void *)-1);
}
