/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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
#include <string.h>
#include <kern/assert.h>
#include <mach/i386/vm_param.h>
#include <mach/i386/kern_return.h>
#include <vm/vm_kern_xnu.h>
#include <vm/pmap.h>
#include "vmx_shims.h"

void *
vmx_pcalloc(void)
{
	char               *pptr;
	kern_return_t   ret;
	ret = kmem_alloc(kernel_map, (vm_offset_t *)&pptr, PAGE_SIZE,
	    KMA_KOBJECT | KMA_DATA | KMA_ZERO, VM_KERN_MEMORY_OSFMK);
	if (ret != KERN_SUCCESS) {
		return NULL;
	}
	return pptr;
}

addr64_t
vmx_paddr(void *va)
{
	return ptoa_64(pmap_find_phys(kernel_pmap, (addr64_t)(uintptr_t)va));
}

void
vmx_pfree(void *va)
{
	kmem_free(kernel_map, (vm_offset_t)va, PAGE_SIZE);
}
