/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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
#include <sys/cdefs.h>

#if __DARWIN_UNIX03

#include <sys/mman.h>
#include <mach/mach_init.h>
#include "stack_logging_internal.h"

/*
 * munmap stub, for stack logging of VM allocations.
 *
 * This is for UNIX03 only.
 */
extern int __munmap(void *, size_t);

int
munmap(void *addr, size_t len)
{
	if (__syscall_logger) {
		__syscall_logger(stack_logging_type_vm_deallocate, (uintptr_t)mach_task_self(), (uintptr_t)addr, len, 0, 0);
	}

	int result = __munmap(addr, len);

	return result;
}

#endif /* __DARWIN_UNIX03 */
