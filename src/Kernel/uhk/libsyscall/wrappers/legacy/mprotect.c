/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 21, 2021.
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

#include "_errno.h"
#include <sys/types.h>
#include <sys/mman.h>
#include <mach/vm_param.h>

/*
 * Stub function to account for the differences in standard compliance
 * while maintaining binary backward compatibility.
 *
 * This is only the legacy behavior.
 */
extern int __mprotect(void *, size_t, int);

int
mprotect(void *addr, size_t len, int prot)
{
	void    *aligned_addr;
	size_t  offset;
	int     rv;

	/*
	 * Page-align "addr" since the system now requires it
	 * for standards compliance.
	 * Update "len" to reflect the alignment.
	 */
	offset = ((uintptr_t) addr) & PAGE_MASK;
	aligned_addr = (void *) (((uintptr_t) addr) & ~PAGE_MASK);
	len += offset;
	rv = __mprotect(aligned_addr, len, prot);
	if (rv == -1 && errno == ENOMEM) {
		/*
		 * Standards now require that we return ENOMEM if there was
		 * a hole in the address range.  Panther and earlier used
		 * to return an EINVAL error, so honor backwards compatibility.
		 */
		errno = EINVAL;
	}
	return rv;
}

#endif /* NO_SYSCALL_LEGACY */
