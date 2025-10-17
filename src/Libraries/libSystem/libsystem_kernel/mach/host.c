/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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
#include <TargetConditionals.h>
#include <machine/cpu_capabilities.h>
#include <mach/kern_return.h>
#include <mach/mach_host.h>
#include <mach/host_priv.h>
#include <sys/types.h>

kern_return_t
host_get_atm_diagnostic_flag(host_t host __unused,
    uint32_t *diagnostic_flag)
{
	*diagnostic_flag = COMM_PAGE_READ(uint32_t, ATM_DIAGNOSTIC_CONFIG);
	return KERN_SUCCESS;
}

kern_return_t
host_get_multiuser_config_flags(host_t host __unused,
    uint32_t *multiuser_flags)
{
#if (TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR)
	*multiuser_flags = COMM_PAGE_READ(uint32_t, MULTIUSER_CONFIG);
	return KERN_SUCCESS;
#else
	(void)multiuser_flags;
	return KERN_NOT_SUPPORTED;
#endif
}

kern_return_t
host_check_multiuser_mode(host_t host __unused,
    uint32_t *multiuser_mode)
{
#if (TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR)
	uint32_t multiuser_flags;
	kern_return_t kr;

	kr = host_get_multiuser_config_flags(host, &multiuser_flags);
	if (kr != KERN_SUCCESS) {
		return kr;
	}
	*multiuser_mode = (multiuser_flags & kIsMultiUserDevice) == kIsMultiUserDevice;
	return KERN_SUCCESS;
#else
	(void)multiuser_mode;
	return KERN_NOT_SUPPORTED;
#endif
}

extern kern_return_t
_kernelrpc_host_create_mach_voucher(mach_port_name_t host,
    mach_voucher_attr_raw_recipe_array_t recipes,
    mach_voucher_attr_recipe_size_t recipesCnt,
    mach_port_name_t *voucher);

kern_return_t
host_create_mach_voucher(mach_port_name_t host,
    mach_voucher_attr_raw_recipe_array_t recipes,
    mach_voucher_attr_recipe_size_t recipesCnt,
    mach_port_name_t *voucher)
{
	kern_return_t rv;

	rv = host_create_mach_voucher_trap(host, recipes, recipesCnt, voucher);

#ifdef __x86_64__
	/* REMOVE once XBS kernel has new trap */
	if (rv == ((1 << 24) | 70)) { /* see mach/i386/syscall_sw.h */
		rv = MACH_SEND_INVALID_DEST;
	}
#elif defined(__i386__)
	/* REMOVE once XBS kernel has new trap */
	if (rv == (kern_return_t)(-70)) {
		rv = MACH_SEND_INVALID_DEST;
	}
#endif

	if (rv == MACH_SEND_INVALID_DEST) {
		rv = _kernelrpc_host_create_mach_voucher(host, recipes, recipesCnt, voucher);
	}

	return rv;
}
