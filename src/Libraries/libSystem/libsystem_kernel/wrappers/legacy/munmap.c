/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 30, 2024.
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
#ifndef NO_SYSCALL_LEGACY

#define _NONSTD_SOURCE
#include <sys/cdefs.h>

#include <sys/types.h>
#include <sys/mman.h>
#include <mach/vm_param.h>
#include <mach/mach_init.h>
#include "stack_logging_internal.h"

/*
 * Stub function to account for the differences in standard compliance
 * while maintaining binary backward compatibility.
 *
 * This is only the legacy behavior.
 */
extern int __munmap(void *, size_t);

int
munmap(void *addr, size_t len)
{
	size_t  offset;

	if (len == 0) {
		/*
		 * Standard compliance now requires the system to return EINVAL
		 * for munmap(addr, 0).  Return success now to maintain
		 * backwards compatibility.
		 */
		return 0;
	}
	/*
	 * Page-align "addr" since the system now requires it
	 * for standards compliance.
	 * Update "len" to reflect the adjustment and still cover the same area.
	 */
	offset = ((uintptr_t) addr) & PAGE_MASK;
	addr = (void *) (((uintptr_t) addr) & ~PAGE_MASK);
	len += offset;

	if (__syscall_logger) {
		__syscall_logger(stack_logging_type_vm_deallocate, (uintptr_t)mach_task_self(), (uintptr_t)addr, len, 0, 0);
	}

	int result = __munmap(addr, len);

	return result;
}
#endif /* NO_SYSCALL_LEGACY */
