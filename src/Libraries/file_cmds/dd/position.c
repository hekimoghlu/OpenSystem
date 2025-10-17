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
#ifndef lint
#if 0
static char sccsid[] = "@(#)position.c	8.3 (Berkeley) 4/2/94";
#endif
#endif /* not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/types.h>

#ifdef __APPLE__
#include <sys/ioctl.h>
#else
#include <sys/mtio.h>
#endif

#include <err.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <unistd.h>

#include "dd.h"
#include "extern.h"

static off_t
seek_offset(IO *io)
{
	off_t n;
	size_t sz;

	n = io->offset;
	sz = io->dbsz;

	_Static_assert(sizeof(io->offset) == sizeof(int64_t), "64-bit off_t");

	/*
	 * If the lseek offset will be negative, verify that this is a special
	 * device file.  Some such files (e.g. /dev/kmem) permit "negative"
	 * offsets.
	 *
	 * Bail out if the calculation of a file offset would overflow.
	 */
	if ((io->flags & ISCHR) == 0 && (n < 0 || n > OFF_MAX / (ssize_t)sz))
		errx(1, "seek offsets cannot be larger than %jd",
		    (intmax_t)OFF_MAX);
	else if ((io->flags & ISCHR) != 0 && (uint64_t)n > UINT64_MAX / sz)
		errx(1, "seek offsets cannot be larger than %ju",
		    (uintmax_t)UINT64_MAX);

	return ((off_t)( (uint64_t)n * sz ));
}

/*
 * Position input/output data streams before starting the copy.  Device type
 * dependent.  Seekable devices use lseek, and the rest position by reading.
 * Seeking past the end of file can cause null blocks to be written to the
 * output.
 */
void
pos_in(void)
{
	off_t cnt;
	int warned;
	ssize_t nr;
	size_t bcnt;

	/* If known to be seekable, try to seek on it. */
	if (in.flags & ISSEEK) {
		errno = 0;
		if (lseek(in.fd, seek_offset(&in), SEEK_CUR) == -1 &&
		    errno != 0)
			err(1, "%s", in.name);
		return;
	}

	/* Don't try to read a really weird amount (like negative). */
	if (in.offset < 0)
		errx(1, "%s: illegal offset", "iseek/skip");

	/*
	 * Read the data.  If a pipe, read until satisfy the number of bytes
	 * being skipped.  No differentiation for reading complete and partial
	 * blocks for other devices.
	 */
	for (bcnt = in.dbsz, cnt = in.offset, warned = 0; cnt;) {
		if ((nr = read(in.fd, in.db, bcnt)) > 0) {
			if (in.flags & ISPIPE) {
				if (!(bcnt -= nr)) {
					bcnt = in.dbsz;
					--cnt;
				}
			} else
				--cnt;
			if (need_summary)
				summary();
			if (need_progress)
				progress();
			continue;
		}

		if (nr == 0) {
			if (files_cnt > 1) {
				--files_cnt;
				continue;
			}
			errx(1, "skip reached end of input");
		}

		/*
		 * Input error -- either EOF with no more files, or I/O error.
		 * If noerror not set die.  POSIX requires that the warning
		 * message be followed by an I/O display.
		 */
		if (ddflags & C_NOERROR) {
			if (!warned) {
				warn("%s", in.name);
				warned = 1;
				summary();
			}
			continue;
		}
		err(1, "%s", in.name);
	}
}

void
pos_out(void)
{
#ifndef __APPLE__
	struct mtop t_op;
#endif
	off_t cnt;
	ssize_t n;

	/*
	 * If not a tape, try seeking on the file.  Seeking on a pipe is
	 * going to fail, but don't protect the user -- they shouldn't
	 * have specified the seek operand.
	 */
	if (out.flags & (ISSEEK | ISPIPE)) {
		errno = 0;
		if (lseek(out.fd, seek_offset(&out), SEEK_CUR) == -1 &&
		    errno != 0)
			err(1, "%s", out.name);
		return;
	}

	/* Don't try to read a really weird amount (like negative). */
	if (out.offset < 0)
		errx(1, "%s: illegal offset", "oseek/seek");

#ifndef __APPLE__
	/* If no read access, try using mtio. */
	if (out.flags & NOREAD) {
		t_op.mt_op = MTFSR;
		t_op.mt_count = out.offset;

		if (ioctl(out.fd, MTIOCTOP, &t_op) == -1)
			err(1, "%s", out.name);
		return;
	}
#endif

	/* Read it. */
	for (cnt = 0; cnt < out.offset; ++cnt) {
		check_terminate();
		if ((n = read(out.fd, out.db, out.dbsz)) > 0)
			continue;
		check_terminate();
		if (n == -1)
			err(1, "%s", out.name);

#ifndef __APPLE__
		/*
		 * If reach EOF, fill with NUL characters; first, back up over
		 * the EOF mark.  Note, cnt has not yet been incremented, so
		 * the EOF read does not count as a seek'd block.
		 */
		t_op.mt_op = MTBSR;
		t_op.mt_count = 1;
		if (ioctl(out.fd, MTIOCTOP, &t_op) == -1)
			err(1, "%s", out.name);
#endif

		while (cnt++ < out.offset) {
			check_terminate();
			n = write(out.fd, out.db, out.dbsz);
			check_terminate();
			if (n == -1)
				err(1, "%s", out.name);
			if (n != out.dbsz)
				errx(1, "%s: write failure", out.name);
		}
		break;
	}
}
