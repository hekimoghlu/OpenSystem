/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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
/*-
 * Copyright (c) 1988 University of Utah.
 * Copyright (c) 1982, 1986, 1990, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * This code is derived from software contributed to Berkeley by
 * the Systems Programming Group of the University of Utah Computer
 * Science Department, and code derived from software contributed to
 * Berkeley by William Jolitz.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * from: Utah $Hdr: mem.c 1.13 89/10/08$
 *	@(#)mem.c	8.1 (Berkeley) 6/11/93
 */

/*
 * Memory special file
 */

#include <sys/param.h>
#include <sys/dir.h>
#include <sys/proc.h>
#include <sys/systm.h>
#include <sys/conf.h>
#include <sys/vm.h>
#include <sys/uio_internal.h>

#include <kern/zalloc.h>

#include <mach/vm_types.h>
#include <mach/vm_param.h>
#include <vm/vm_kern.h>         /* for kernel_map */
#include <libkern/section_keywords.h>

#include <pexpert/pexpert.h>    /* for PE_parse_boot_argn */

boolean_t iskmemdev(dev_t dev);

#if CONFIG_DEV_KMEM
boolean_t dev_kmem_enabled;
boolean_t dev_kmem_mask_top_bit;

void dev_kmem_init(void);

#if defined(__x86_64__)
extern addr64_t  kvtophys(vm_offset_t va);
#else
#error need kvtophys prototype
#endif
extern boolean_t kernacc(off_t, size_t );

#endif

static SECURITY_READ_ONLY_LATE(caddr_t) devzerobuf;

int mmread(dev_t dev, struct uio *uio);
int mmwrite(dev_t dev, struct uio *uio);
int mmioctl(dev_t dev, u_long cmd, caddr_t data, int flag, struct proc *p);
int mmrw(dev_t dev, struct uio *uio, enum uio_rw rw);

int
mmread(dev_t dev, struct uio *uio)
{
	return mmrw(dev, uio, UIO_READ);
}

int
mmwrite(dev_t dev, struct uio *uio)
{
	return mmrw(dev, uio, UIO_WRITE);
}

int
mmioctl(dev_t dev, u_long cmd, __unused caddr_t data,
    __unused int flag, __unused struct proc *p)
{
	int minnum = minor(dev);

	if (0 == minnum || 1 == minnum) {
		/* /dev/mem and /dev/kmem */
#if CONFIG_DEV_KMEM
		if (!dev_kmem_enabled) {
			return ENODEV;
		}
#else
		return ENODEV;
#endif
	}

	switch (cmd) {
	case FIONBIO:
	case FIOASYNC:
		/* OK to do nothing: we always return immediately */
		break;
	default:
		return ENODEV;
	}

	return 0;
}

int
mmrw(dev_t dev, struct uio *uio, enum uio_rw rw)
{
	user_size_t c;
	int error = 0;

	while (uio_resid(uio) > 0) {
		uio_update(uio, 0);

		switch (minor(dev)) {
		/* minor device 0 is physical memory */
		case 0:
			return ENODEV;

		/* minor device 1 is kernel memory */
		case 1:
#if !CONFIG_DEV_KMEM
			return ENODEV;
#else /* CONFIG_DEV_KMEM */
			if (!dev_kmem_enabled) {
				return ENODEV;
			}

			vm_address_t kaddr = (vm_address_t)uio->uio_offset;
			if (dev_kmem_mask_top_bit) {
				/*
				 * KVA addresses of the form 0xFFFFFF80AABBCCDD can't be
				 * represented as a signed off_t correctly. In these cases,
				 * 0x7FFFFF80AABBCCDD is passed in, and the top bit OR-ed
				 * on.
				 */
				const vm_address_t top_bit = (~((vm_address_t)0)) ^ (~((vm_address_t)0) >> 1UL);
				if (kaddr & top_bit) {
					/* top bit should not be set already */
					return EFAULT;
				}
				kaddr |= top_bit;
			}

			c = uio_curriovlen(uio);

			/* Do some sanity checking */
			if ((kaddr > (VM_MAX_KERNEL_ADDRESS - c)) ||
			    (kaddr <= VM_MIN_KERNEL_AND_KEXT_ADDRESS)) {
				goto fault;
			}
			if (!kernacc(kaddr, c)) {
				goto fault;
			}
			error = uiomove((const char *)(uintptr_t)kaddr,
			    (int)c, uio);
			if (error) {
				break;
			}

			continue; /* Keep going until UIO is done */
#endif /* CONFIG_DEV_KMEM */

		/* minor device 2 is EOF/RATHOLE */
		case 2:
			if (rw == UIO_READ) {
				return 0;
			}
			c = uio_curriovlen(uio);

			error = 0; /* Always succeeds, always consumes all input */
			break;
		case 3:
			assert(devzerobuf != NULL);

			if (uio->uio_rw == UIO_WRITE) {
				c = uio_curriovlen(uio);

				error = 0; /* Always succeeds, always consumes all input */
				break;
			}

			c = MIN(uio_curriovlen(uio), PAGE_SIZE);
			error = uiomove(devzerobuf, (int)c, uio);
			if (error) {
				break;
			}

			continue; /* Keep going until UIO is done */
		default:
			return ENODEV;
		}

		if (error) {
			break;
		}

		uio_update(uio, c);
	}
	return error;
#if CONFIG_DEV_KMEM
fault:
	return EFAULT;
#endif
}

__startup_func
static void
devzerobuf_init(void)
{
	devzerobuf = zalloc_permanent(PAGE_SIZE, ZALIGN_NONE); /* zeroed */
}
STARTUP(ZALLOC, STARTUP_RANK_LAST, devzerobuf_init);

#if CONFIG_DEV_KMEM
void
dev_kmem_init(void)
{
	uint32_t kmem;

	if (PE_i_can_has_debugger(NULL) &&
	    PE_parse_boot_argn("kmem", &kmem, sizeof(kmem))) {
		if (kmem & 0x1) {
			dev_kmem_enabled = TRUE;
		}
		if (kmem & 0x2) {
			dev_kmem_mask_top_bit = TRUE;
		}
	}
}

boolean_t
kernacc(
	off_t       start,
	size_t      len
	)
{
	off_t base;
	off_t end;

	base = trunc_page(start);
	end = start + len;

	while (base < end) {
		if (kvtophys((vm_offset_t)base) == 0ULL) {
			return FALSE;
		}
		base += page_size;
	}

	return TRUE;
}

#endif /* CONFIG_DEV_KMEM */

/*
 * Returns true if dev is /dev/mem or /dev/kmem.
 */
boolean_t
iskmemdev(dev_t dev)
{
	return major(dev) == 3 && minor(dev) < 2;
}
