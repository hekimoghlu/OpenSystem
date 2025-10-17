/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <err.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef __APPLE__
#include <libcasper.h>
#include <casper/cap_fileargs.h>
#endif

#include "extern.h"

void
ierr(const char *fname)
{
	warn("%s", fname);
	rval = 1;
}

void
oerr(void)
{
	err(1, "stdout");
}

/*
 * Print `len' bytes from the file associated with `mip', starting at
 * absolute file offset `startoff'. May move map window.
 */
int
mapprint(struct mapinfo *mip, off_t startoff, off_t len)
{
	int n;

	while (len > 0) {
		if (startoff < mip->mapoff || startoff >= mip->mapoff +
		    (off_t)mip->maplen) {
			if (maparound(mip, startoff) != 0)
				return (1);
		}
		n = (mip->mapoff + mip->maplen) - startoff;
		if (n > len)
			n = len;
		WR(mip->start + (startoff - mip->mapoff), n);
		startoff += n;
		len -= n;
	}
	return (0);
}

/*
 * Move the map window so that it contains the byte at absolute file
 * offset `offset'. The start of the map window will be TAILMAPLEN
 * aligned.
 */
int
maparound(struct mapinfo *mip, off_t offset)
{

	if (mip->start != NULL && munmap(mip->start, mip->maplen) != 0)
		return (1);

	mip->mapoff = offset & ~((off_t)TAILMAPLEN - 1);
	mip->maplen = TAILMAPLEN;
	if ((off_t)mip->maplen > mip->maxoff - mip->mapoff)
		mip->maplen = mip->maxoff - mip->mapoff;
	if (mip->maplen <= 0)
		abort();
	if ((mip->start = mmap(NULL, mip->maplen, PROT_READ, MAP_SHARED,
	     mip->fd, mip->mapoff)) == MAP_FAILED)
		return (1);

	return (0);
}

/*
 * Print the file name without stdio buffering.
 */
void
printfn(const char *fn, int print_nl)
{

	if (print_nl)
		WR("\n", 1);
	WR("==> ", 4);
	WR(fn, strlen(fn));
	WR(" <==\n", 5);
}
