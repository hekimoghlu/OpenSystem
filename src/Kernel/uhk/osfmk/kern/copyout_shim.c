/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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
#include <mach/mach_types.h>
#include <mach/vm_param.h>
#include <string.h>
#include <pexpert/pexpert.h>
#include <kern/copyout_shim.h>

#if (DEVELOPMENT || DEBUG)
#define UNUSED_IN_RELEASE(x)
#else
//supress compiler warnings about unused variables
#define UNUSED_IN_RELEASE(x) (void)(x)
#endif /* (DEVELOPMENT || DEBUG) */


#if (DEVELOPMENT || DEBUG)
copyout_shim_fn_t copyout_shim_fn = NULL;
unsigned co_src_flags = 0;
#endif

kern_return_t
register_copyout_shim(void (*fn)(const void *, user_addr_t, vm_size_t, unsigned co_src), unsigned types)
{
#if (DEVELOPMENT || DEBUG)
	int copyout_shim_enabled = 0;

	if (!fn) {
		/* unregistration is always allowed */
		copyout_shim_fn = NULL;
		return KERN_SUCCESS;
	}

	if (copyout_shim_fn) {
		//need to unregister first before registering a new one.
		return KERN_FAILURE;
	}

	if (!PE_parse_boot_argn("enable_copyout_shim", &copyout_shim_enabled, sizeof(copyout_shim_enabled)) || !copyout_shim_enabled) {
		return KERN_FAILURE;
	}


	co_src_flags = types;
	copyout_shim_fn = fn;
	return KERN_SUCCESS;
#else
	UNUSED_IN_RELEASE(fn);
	UNUSED_IN_RELEASE(types);
	return KERN_FAILURE;
#endif
}

void *
cos_kernel_unslide(const void *ptr)
{
#if (DEVELOPMENT || DEBUG)
	return (void *)(VM_KERNEL_UNSLIDE(ptr));
#else
	UNUSED_IN_RELEASE(ptr);
	return NULL;
#endif
}

void *
cos_kernel_reslide(const void *ptr)
{
#if (DEVELOPMENT || DEBUG)
	return (void *)(VM_KERNEL_SLIDE(ptr));
#else
	UNUSED_IN_RELEASE(ptr);
	return NULL;
#endif
}
