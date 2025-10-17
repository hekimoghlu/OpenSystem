/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 20, 2024.
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
#ifdef __APPLE__
#include <sys/cdefs.h>
static const char rcsid[] __used =
#else
static const char rcsid[] =
#endif
    "$FreeBSD$";

#include <sys/stat.h>

#include <ctype.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <libutil.h>

static void	usage(void);

int
main(int argc, char **argv)
{
	struct stat sb;
	mode_t omode;
#ifdef __APPLE__
	off_t oflow, rsize, sz, tsize, round;
#else
	off_t oflow, rsize, sz, tsize, round, off, len;
#endif
	uint64_t usz;
	int ch, error, fd, oflags, r;
#ifndef __APPLE__
	int do_dealloc;
#endif
	int do_truncate;
	int no_create;
	int do_relative;
	int do_round;
	int do_refer;
	int got_size;
	char *fname, *rname;
#ifndef __APPLE__
	struct spacectl_range sr;
#endif

	fd = -1;
#ifdef __APPLE__
	rsize = tsize = sz = 0;
	no_create = do_relative = do_round = do_refer =
	    got_size = 0;
#else
	rsize = tsize = sz = off = 0;
	len = -1;
	do_dealloc = no_create = do_relative = do_round = do_refer =
	    got_size = 0;
#endif
	do_truncate = 1;
	error = r = 0;
	rname = NULL;
#ifdef __APPLE__
	while ((ch = getopt(argc, argv, "cr:s:")) != -1)
#else
	while ((ch = getopt(argc, argv, "cdr:s:o:l:")) != -1)
#endif
		switch (ch) {
		case 'c':
			no_create = 1;
			break;
#ifndef __APPLE__
		case 'd':
			do_dealloc = 1;
			do_truncate = 0;
			break;
#endif
		case 'r':
			do_refer = 1;
			rname = optarg;
			break;
		case 's':
			if (*optarg == '+' || *optarg == '-') {
				do_relative = 1;
			} else if (*optarg == '%' || *optarg == '/') {
				do_round = 1;
			}
			if (expand_number(do_relative || do_round ?
			    optarg + 1 : optarg,
			    &usz) == -1 || (off_t)usz < 0)
				errx(EXIT_FAILURE,
				    "invalid size argument `%s'", optarg);

			sz = (*optarg == '-' || *optarg == '/') ?
				-(off_t)usz : (off_t)usz;
			got_size = 1;
			break;
#ifndef __APPLE__
		case 'o':
			if (expand_number(optarg, &usz) == -1 ||
			    (off_t)usz < 0)
				errx(EXIT_FAILURE,
				    "invalid offset argument `%s'", optarg);

			off = usz;
			break;
		case 'l':
			if (expand_number(optarg, &usz) == -1 ||
			    (off_t)usz <= 0)
				errx(EXIT_FAILURE,
				    "invalid length argument `%s'", optarg);

			len = usz;
			break;
#endif
		default:
			usage();
			/* NOTREACHED */
		}

	argv += optind;
	argc -= optind;

	/*
	 * Exactly one of do_refer, got_size or do_dealloc must be specified.
	 * Since do_relative implies got_size, do_relative, do_refer and
	 * do_dealloc are also mutually exclusive.  If do_dealloc is specified,
	 * the length argument must be set.  See usage() for allowed
	 * invocations.
	 */
#ifdef __APPLE__
	if (argc < 1 || do_refer + got_size != 1)
#else
	if (argc < 1 || do_refer + got_size + do_dealloc != 1 ||
	    (do_dealloc == 1 && len == -1))
#endif
		usage();
	if (do_refer == 1) {
		if (stat(rname, &sb) == -1)
			err(EXIT_FAILURE, "%s", rname);
		tsize = sb.st_size;
	} else if (do_relative == 1 || do_round == 1)
		rsize = sz;
#ifdef __APPLE__
	else
#else
	else if (do_dealloc == 0)
#endif
		tsize = sz;

	if (no_create)
		oflags = O_WRONLY;
	else
		oflags = O_WRONLY | O_CREAT;
	omode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

	while ((fname = *argv++) != NULL) {
		if (fd != -1)
			close(fd);
		if ((fd = open(fname, oflags, omode)) == -1) {
			if (errno != ENOENT) {
				warn("%s", fname);
				error++;
			}
			continue;
		}
		if (do_relative == 1) {
			if (fstat(fd, &sb) == -1) {
				warn("%s", fname);
				error++;
				continue;
			}
			oflow = sb.st_size + rsize;
			if (oflow < (sb.st_size + rsize)) {
				errno = EFBIG;
				warn("%s", fname);
				error++;
				continue;
			}
			tsize = oflow;
		}
		if (do_round == 1) {
			if (fstat(fd, &sb) == -1) {
				warn("%s", fname);
				error++;
				continue;
			}
			sz = rsize;
			if (sz < 0)
				sz = -sz;
			if (sb.st_size % sz) {
				round = sb.st_size / sz;
				if (round != sz && rsize < 0)
					round--;
				else if (rsize > 0)
					round++;
				tsize = (round < 0 ? 0 : round) * sz;
			} else
				tsize = sb.st_size;
		}
		if (tsize < 0)
			tsize = 0;

#ifndef __APPLE__
		if (do_dealloc == 1) {
			sr.r_offset = off;
			sr.r_len = len;
			r = fspacectl(fd, SPACECTL_DEALLOC, &sr, 0, &sr);
		}
#endif
		if (do_truncate == 1)
			r = ftruncate(fd, tsize);
		if (r == -1) {
			warn("%s", fname);
			error++;
		}
	}
	if (fd != -1)
		close(fd);

	return (error ? EXIT_FAILURE : EXIT_SUCCESS);
}

static void
usage(void)
{
#ifdef __APPLE__
	fprintf(stderr, "%s\n%s\n",
	    "usage: truncate [-c] -s [+|-|%|/]size[K|k|M|m|G|g|T|t] file ...",
	    "       truncate [-c] -r rfile file ...");
#else
	fprintf(stderr, "%s\n%s\n%s\n",
	    "usage: truncate [-c] -s [+|-|%|/]size[K|k|M|m|G|g|T|t] file ...",
	    "       truncate [-c] -r rfile file ...",
	    "       truncate [-c] -d [-o offset[K|k|M|m|G|g|T|t]] -l length[K|k|M|m|G|g|T|t] file ...");
#endif
	exit(EXIT_FAILURE);
}
