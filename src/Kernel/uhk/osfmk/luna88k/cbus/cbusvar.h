/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
 * Copyright (c) 2014 Kenji Aoyama.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

/*
 * PC-9801 extension board slot bus ('C-bus') driver for LUNA-88K2.
 */

#include <sys/evcount.h>
#include <sys/queue.h>

#include <arch/luna88k/include/board.h>

#define	PCEXMEM_BASE	PC_BASE
#define	PCEXIO_BASE	(PC_BASE + 0x1000000)

/*
 * Currently 7 level C-bus interrupts (INT0 - INT6) are supported.
 */
#define NCBUSISR	7

/*
 * C-bus interrupt handler
 */
struct cbus_isr_t {
	int		(*isr_func)(void *);
	void		*isr_arg;
	int		isr_intlevel;
	int		isr_ipl;
	struct evcount	isr_count;
};

int	cbus_isrlink(int (*)(void *), void *, int, int, const char *);
int	cbus_isrunlink(int (*)(void *), int);
u_int8_t	cbus_intr_registered(void);

struct cbus_attach_args {
	char		*ca_name;
	u_int32_t	ca_iobase;
	u_int32_t	ca_iosize;
	u_int32_t	ca_maddr;
	u_int32_t	ca_msize;
	u_int32_t	ca_int;
};

#define	cf_iobase	cf_loc[0]
#define	cf_iosize	cf_loc[1]
#define	cf_maddr	cf_loc[2]
#define	cf_msize	cf_loc[3]
#define	cf_int		cf_loc[4]
