/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
 * Copyright (c) 2010 Miodrag Vallat.
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

#include <sys/param.h>
#include <sys/reboot.h>
#include <machine/cpu.h>
#include <machine/pmon.h>
#include "libsa.h"
#include <lib/libsa/loadfile.h>

typedef void (*program)(int32_t, int32_t, int32_t *, int32_t, uint64_t *);

#define	PTR_TO_CKSEG1(ptr)	(int32_t)(CKSEG1_BASE | (uint64_t)(ptr))

void
run_loadfile(uint64_t *marks, int howto)
{
	int32_t newargc;
	int32_t *newargv;
	char kernelflags[8];
	char *c;
	const char *arg;

	/*
	 * Build a new commandline:
	 * boot <device kernel is loaded from> -<kernel options>
	 */

	newargc = howto == 0 ? 2 : 3;
	newargv = alloc(newargc * sizeof(int32_t));
	if (newargv == NULL)
		panic("out of memory");

	arg = "boot";	/* kernel needs this. */
	newargv[0] = PTR_TO_CKSEG1(arg);
	newargv[1] = PTR_TO_CKSEG1(&pmon_bootdev);
	if (howto != 0) {
		c = kernelflags;
		*c++ = '-';
		if (howto & RB_ASKNAME)
			*c++ = 'a';
		if (howto & RB_CONFIG)
			*c++ = 'c';
		if (howto & RB_KDB)
			*c++ = 'd';
		if (howto & RB_GOODRANDOM)
			*c++ = 'g';
		if (howto & RB_SINGLE)
			*c++ = 's';
		*c = '\0';
		newargv[2] = PTR_TO_CKSEG1(&kernelflags);
	}

	pmon_cacheflush();

	(*(program)(marks[MARK_ENTRY]))(newargc, PTR_TO_CKSEG1(newargv),
	    pmon_envp, pmon_callvec,
	    (uint64_t *)PHYS_TO_CKSEG0(marks[MARK_END]));

	rd_invalidate();
	_rtt();
}

