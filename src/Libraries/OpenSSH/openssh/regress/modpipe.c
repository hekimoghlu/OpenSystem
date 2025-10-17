/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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
/* $OpenBSD: modpipe.c,v 1.6 2013/11/21 03:16:47 djm Exp $ */

#include "includes.h"

#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <errno.h>
#ifdef HAVE_ERR_H
# include <err.h>
#endif
#include "openbsd-compat/getopt_long.c"

static void
usage(void)
{
	fprintf(stderr, "Usage: modpipe -w [-m modspec ...] < in > out\n");
	fprintf(stderr, "modspec is one of:\n");
	fprintf(stderr, "    xor:offset:value       - XOR \"value\" at \"offset\"\n");
	fprintf(stderr, "    andor:offset:val1:val2 - AND \"val1\" then OR \"val2\" at \"offset\"\n");
	exit(1);
}

#define MAX_MODIFICATIONS 256
struct modification {
	enum { MOD_XOR, MOD_AND_OR } what;
	unsigned long long offset;
	u_int8_t m1, m2;
};

static void
parse_modification(const char *s, struct modification *m)
{
	char what[16+1];
	int n, m1, m2;

	bzero(m, sizeof(*m));
	if ((n = sscanf(s, "%16[^:]%*[:]%llu%*[:]%i%*[:]%i",
	    what, &m->offset, &m1, &m2)) < 3)
		errx(1, "Invalid modification spec \"%s\"", s);
	if (strcasecmp(what, "xor") == 0) {
		if (n > 3)
			errx(1, "Invalid modification spec \"%s\"", s);
		if (m1 < 0 || m1 > 0xff)
			errx(1, "Invalid XOR modification value");
		m->what = MOD_XOR;
		m->m1 = m1;
	} else if (strcasecmp(what, "andor") == 0) {
		if (n != 4)
			errx(1, "Invalid modification spec \"%s\"", s);
		if (m1 < 0 || m1 > 0xff)
			errx(1, "Invalid AND modification value");
		if (m2 < 0 || m2 > 0xff)
			errx(1, "Invalid OR modification value");
		m->what = MOD_AND_OR;
		m->m1 = m1;
		m->m2 = m2;
	} else
		errx(1, "Invalid modification type \"%s\"", what);
}

int
main(int argc, char **argv)
{
	int ch;
	u_char buf[8192];
	size_t total;
	ssize_t r, s, o;
	struct modification mods[MAX_MODIFICATIONS];
	u_int i, wflag = 0, num_mods = 0;

	while ((ch = getopt(argc, argv, "wm:")) != -1) {
		switch (ch) {
		case 'm':
			if (num_mods >= MAX_MODIFICATIONS)
				errx(1, "Too many modifications");
			parse_modification(optarg, &(mods[num_mods++]));
			break;
		case 'w':
			wflag = 1;
			break;
		default:
			usage();
			/* NOTREACHED */
		}
	}
	for (total = 0;;) {
		r = s = read(STDIN_FILENO, buf, sizeof(buf));
		if (r == 0)
			break;
		if (r < 0) {
			if (errno == EAGAIN || errno == EINTR)
				continue;
			err(1, "read");
		}
		for (i = 0; i < num_mods; i++) {
			if (mods[i].offset < total ||
			    mods[i].offset >= total + s)
				continue;
			switch (mods[i].what) {
			case MOD_XOR:
				buf[mods[i].offset - total] ^= mods[i].m1;
				break;
			case MOD_AND_OR:
				buf[mods[i].offset - total] &= mods[i].m1;
				buf[mods[i].offset - total] |= mods[i].m2;
				break;
			}
		}
		for (o = 0; o < s; o += r) {
			r = write(STDOUT_FILENO, buf, s - o);
			if (r == 0)
				break;
			if (r < 0) {
				if (errno == EAGAIN || errno == EINTR)
					continue;
				err(1, "write");
			}
		}
		total += s;
	}
	/* Warn if modifications not reached in input stream */
	r = 0;
	for (i = 0; wflag && i < num_mods; i++) {
		if (mods[i].offset < total)
			continue;
		r = 1;
		fprintf(stderr, "modpipe: warning - mod %u not reached\n", i);
	}
	return r;
}
