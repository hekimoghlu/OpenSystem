/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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
 * Mach Operating System
 * Copyright (c) 1992 Carnegie Mellon University
 * All Rights Reserved.
 * 
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 * 
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 * 
 * Carnegie Mellon requests users of this software to return to
 * 
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 * 
 * any improvements or extensions that they make and grant Carnegie Mellon 
 * the rights to redistribute these changes.
 */

/*
 * Interface to the debugger for virtual memory read/write.
 * This is a simple version for kernels with writable text.
 * For an example of read-only kernel text, see the file:
 * sys/arch/sun3/sun3/db_memrw.c
 *
 * ALERT!  If you want to access device registers with a
 * specific size, then the read/write functions have to
 * make sure to do the correct sized pointer access.
 */

#include <sys/param.h>
#include <sys/proc.h>
#include <sys/systm.h>

#include <uvm/uvm_extern.h>

#include <machine/db_machdep.h>
#include <machine/pcb.h>

#include <ddb/db_access.h>

/*
 * Read bytes from kernel address space for debugger.
 */
void
db_read_bytes(vaddr_t addr, size_t size, void *datap)
{
	char *data = datap, *src = (char *)addr;
	faultbuf env;
	faultbuf *old_onfault = curpcb->pcb_onfault;
	if (setfault(&env)) {
		curpcb->pcb_onfault = old_onfault;
		return;
	}

	if (size == 4) {
		*((int *)data) = *((int *)src);
	} else if (size == 2) {
		*((short *)data) = *((short *)src);
	} else {
		while (size > 0) {
			--size;
			*data++ = *src++;
		}
	}
	curpcb->pcb_onfault = old_onfault;
}

/*
 * Write bytes to kernel address space for debugger.
 */
void
db_write_bytes(vaddr_t addr, size_t size, void *datap)
{
	char *data = datap, *dst = (char *)addr;
	faultbuf env;
	faultbuf *old_onfault = curpcb->pcb_onfault;

	if (setfault(&env)) {
		curpcb->pcb_onfault = old_onfault;
		return;
	}

	if (size == 4) {
		*((int *)dst) = *((int *)data);
	} else if (size == 2) {
		*((short *)dst) = *((short *)data);
	} else  {
		while (size > 0) {
			--size;
			*dst++ = *data++;
		}
	}
	syncicache((void *)addr, size);
	curpcb->pcb_onfault = old_onfault;
}


