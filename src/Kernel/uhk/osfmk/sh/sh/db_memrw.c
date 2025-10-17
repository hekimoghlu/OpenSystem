/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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
 * Copyright (c) 1991,1990 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS
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
 * any improvements or extensions that they make and grant Carnegie the
 * rights to redistribute these changes.
 *
 *	db_interface.c,v 2.4 1991/02/05 17:11:13 mrt (CMU)
 */

/*
 * Routines to read and write memory on behalf of the debugger, used
 * by DDB.
 */

#include <sys/param.h>
#include <sys/proc.h>
#include <sys/systm.h>
#include <sys/stdint.h>

#include <uvm/uvm_extern.h>

#include <machine/db_machdep.h>

#include <ddb/db_access.h>

/*
 * Read bytes from kernel address space for debugger.
 */
void
db_read_bytes(vaddr_t addr, size_t size, void *datap)
{
	char *data = datap, *src = (char *)addr;

	/* properly aligned 4-byte */
	if (size == 4 && ((addr & 3) == 0) && (((uintptr_t)data & 3) == 0)) {
		*(uint32_t *)data = *(uint32_t *)src;
		return;
	}

	/* properly aligned 2-byte */
	if (size == 2 && ((addr & 1) == 0) && (((uintptr_t)data & 1) == 0)) {
		*(uint16_t *)data = *(uint16_t *)src;
		return;
	}

	while (size-- > 0)
		*data++ = *src++;
}

/*
 * Write bytes to kernel address space for debugger.
 */
void
db_write_bytes(vaddr_t addr, size_t size, void *datap)
{
	char *data = datap, *dst = (char *)addr;

	/* properly aligned 4-byte */
	if (size == 4 && ((addr & 3) == 0) && (((uintptr_t)data & 3) == 0)) {
		*(uint32_t *)dst = *(const uint32_t *)data;
		return;
	}

	/* properly aligned 2-byte */
	if (size == 2 && ((addr & 1) == 0) && (((uintptr_t)data & 1) == 0)) {
		*(uint16_t *)dst = *(const uint16_t *)data;
		return;
	}

	while (size-- > 0)
		*dst++ = *data++;
}

