/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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
#include <mach/vm_param.h>
#include <errno.h>
#include <mach/mach_init.h>
#include "stack_logging_internal.h"

void *__mmap(void *addr, size_t len, int prot, int flags, int fildes, off_t off);

/*
 * mmap stub, with preemptory failures due to extra parameter checking
 * mandated for conformance.
 *
 * This is for UNIX03 only.
 */
void *
mmap(void *addr, size_t len, int prot, int flags, int fildes, off_t off)
{
	/*
	 * Preemptory failures:
	 *
	 * o	off is not a multiple of the page size
	 *      [ This is enforced by the kernel with MAP_UNIX03 ]
	 * o	flags does not contain either MAP_PRIVATE or MAP_SHARED
	 * o	len is zero
	 *
	 * Now enforced by the kernel when the MAP_UNIX03 flag is provided.
	 */
	extern void cerror_nocancel(int);
	if ((((flags & MAP_PRIVATE) != MAP_PRIVATE) &&
	    ((flags & MAP_SHARED) != MAP_SHARED)) ||
	    (len == 0)) {
		cerror_nocancel(EINVAL);
		return MAP_FAILED;
	}

	void *ptr = __mmap(addr, len, prot, flags | MAP_UNIX03, fildes, off);

	if (__syscall_logger) {
		int stackLoggingFlags = stack_logging_type_vm_allocate;
		if (flags & MAP_ANON) {
			stackLoggingFlags |= (fildes & VM_FLAGS_ALIAS_MASK);
		} else {
			stackLoggingFlags |= stack_logging_type_mapped_file_or_shared_mem;
		}
		__syscall_logger(stackLoggingFlags, (uintptr_t)mach_task_self(), (uintptr_t)len, 0, (uintptr_t)ptr, 0);
	}

	return ptr;
}

#endif /* __DARWIN_UNIX03 */
