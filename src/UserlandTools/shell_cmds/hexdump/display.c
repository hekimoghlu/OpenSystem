/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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
static char sccsid[] = "@(#)display.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/param.h>
#ifndef __APPLE__
#include <sys/capsicum.h>
#endif
#include <sys/conf.h>
#include <sys/ioctl.h>
#include <sys/stat.h>

#ifndef __APPLE__
#include <capsicum_helpers.h>
#endif
#include <ctype.h>
#include <err.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "hexdump.h"

enum _vflag vflag = FIRST;

static off_t address;			/* address/offset in stream */
static off_t eaddress;			/* end address */

static void print(PR *, u_char *);
static void noseek(void);

void
display(void)
{
	FS *fs;
	FU *fu;
	PR *pr;
	int cnt;
	u_char *bp;
	off_t saveaddress;
	u_char savech, *savebp;

	savech = 0;
	while ((bp = get()))
	    for (fs = fshead, savebp = bp, saveaddress = address; fs;
		fs = fs->nextfs, bp = savebp, address = saveaddress)
		    for (fu = fs->nextfu; fu; fu = fu->nextfu) {
			if (fu->flags&F_IGNORE)
				break;
			for (cnt = fu->reps; cnt; --cnt)
			    for (pr = fu->nextpr; pr; address += pr->bcnt,
				bp += pr->bcnt, pr = pr->nextpr) {
				    if (eaddress && address >= eaddress &&
					!(pr->flags & (F_TEXT|F_BPAD)))
					    bpad(pr);
				    if (cnt == 1 && pr->nospace) {
					savech = *pr->nospace;
					*pr->nospace = '\0';
				    }
				    print(pr, bp);
				    if (cnt == 1 && pr->nospace)
					*pr->nospace = savech;
			    }
		    }
	if (endfu) {
		/*
		 * If eaddress not set, error or file size was multiple of
		 * blocksize, and no partial block ever found.
		 */
		if (!eaddress) {
			if (!address)
				return;
			eaddress = address;
		}
/* rdar://84571853 (Enable -Wformat-nonliteral compiler flag in shell_cmds)
 * Here, the whole point of this program is to allow the user to specify a custom
 * format, so we just suppress the warning.
 */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
		for (pr = endfu->nextpr; pr; pr = pr->nextpr)
			switch(pr->flags) {
			case F_ADDRESS:
				(void)printf(pr->fmt, (quad_t)eaddress);
				break;
			case F_TEXT:
				(void)printf("%s", pr->fmt);
				break;
			}
#pragma clang diagnostic pop
	}
}

static void
print(PR *pr, u_char *bp)
{
	long double ldbl;
	   double f8;
	    float f4;
	  int16_t s2;
	   int8_t s8;
	  int32_t s4;
	u_int16_t u2;
	u_int32_t u4;
	u_int64_t u8;

/* rdar://84571853 (Enable -Wformat-nonliteral compiler flag in shell_cmds)
 * Here, the whole point of this program is to allow the user to specify a custom
 * format, so we just suppress the warning.
 */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
	switch(pr->flags) {
	case F_ADDRESS:
		(void)printf(pr->fmt, (quad_t)address);
		break;
	case F_BPAD:
		(void)printf(pr->fmt, "");
		break;
	case F_C:
		conv_c(pr, bp, eaddress ? eaddress - address :
		    blocksize - address % blocksize);
		break;
	case F_CHAR:
		(void)printf(pr->fmt, *bp);
		break;
	case F_DBL:
		switch(pr->bcnt) {
		case 4:
			bcopy(bp, &f4, sizeof(f4));
			(void)printf(pr->fmt, f4);
			break;
		case 8:
			bcopy(bp, &f8, sizeof(f8));
			(void)printf(pr->fmt, f8);
			break;
		default:
			if (pr->bcnt == sizeof(long double)) {
				bcopy(bp, &ldbl, sizeof(ldbl));
				(void)printf(pr->fmt, ldbl);
			}
			break;
		}
		break;
	case F_INT:
		switch(pr->bcnt) {
		case 1:
			(void)printf(pr->fmt, (quad_t)(signed char)*bp);
			break;
		case 2:
			bcopy(bp, &s2, sizeof(s2));
			(void)printf(pr->fmt, (quad_t)s2);
			break;
		case 4:
			bcopy(bp, &s4, sizeof(s4));
			(void)printf(pr->fmt, (quad_t)s4);
			break;
		case 8:
			bcopy(bp, &s8, sizeof(s8));
			(void)printf(pr->fmt, s8);
			break;
		}
		break;
	case F_P:
#ifdef __APPLE__ /* rdar://91259683 */
		(void)printf(pr->fmt, isprint(*bp) && isascii(*bp) ? *bp : '.');
#else
		(void)printf(pr->fmt, isprint(*bp) ? *bp : '.');
#endif /* __APPLE__ */
		break;
	case F_STR:
		(void)printf(pr->fmt, (char *)bp);
		break;
	case F_TEXT:
		(void)printf("%s", pr->fmt);
		break;
	case F_U:
		conv_u(pr, bp);
		break;
	case F_UINT:
		switch(pr->bcnt) {
		case 1:
			(void)printf(pr->fmt, (u_quad_t)*bp);
			break;
		case 2:
			bcopy(bp, &u2, sizeof(u2));
			(void)printf(pr->fmt, (u_quad_t)u2);
			break;
		case 4:
			bcopy(bp, &u4, sizeof(u4));
			(void)printf(pr->fmt, (u_quad_t)u4);
			break;
		case 8:
			bcopy(bp, &u8, sizeof(u8));
			(void)printf(pr->fmt, u8);
			break;
		}
		break;
	}
#pragma clang diagnostic pop
}

void
bpad(PR *pr)
{
	static char const *spec = " -0+#";
	char *p1, *p2;

	/*
	 * Remove all conversion flags; '-' is the only one valid
	 * with %s, and it's not useful here.
	 */
	pr->flags = F_BPAD;
	pr->cchar[0] = 's';
	pr->cchar[1] = '\0';
	for (p1 = pr->fmt; *p1 != '%'; ++p1);
	for (p2 = ++p1; *p1 && strchr(spec, *p1); ++p1);
	while ((*p2++ = *p1++));
}

static char **_argv;

u_char *
get(void)
{
	static int ateof = 1;
	static u_char *curp, *savp;
	int n;
	int need, nread;
	int valid_save = 0;
	u_char *tmpp;

	if (!curp) {
		if ((curp = calloc(1, blocksize)) == NULL)
			err(1, NULL);
		if ((savp = calloc(1, blocksize)) == NULL)
			err(1, NULL);
	} else {
		tmpp = curp;
		curp = savp;
		savp = tmpp;
		address += blocksize;
		valid_save = 1;
	}
	for (need = blocksize, nread = 0;;) {
		/*
		 * if read the right number of bytes, or at EOF for one file,
		 * and no other files are available, zero-pad the rest of the
		 * block and set the end flag.
		 */
		if (!length || (ateof && !next((char **)NULL))) {
			if (odmode && address < skip)
				errx(1, "cannot skip past end of input");
			if (need == blocksize)
				return((u_char *)NULL);
			/*
			 * XXX bcmp() is not quite right in the presence
			 * of multibyte characters.
			 */
#ifdef __APPLE__
			/* 5650060 */
			if (!need && vflag != ALL &&
#else
			if (vflag != ALL &&
#endif
			    valid_save &&
			    bcmp(curp, savp, nread) == 0) {
				if (vflag != DUP)
					(void)printf("*\n");
				return((u_char *)NULL);
			}
			bzero((char *)curp + nread, need);
			eaddress = address + nread;
			return(curp);
		}
		n = fread((char *)curp + nread, sizeof(u_char),
		    length == -1 ? need : MIN(length, need), stdin);
		if (!n) {
			if (ferror(stdin))
				warn("%s", _argv[-1]);
			ateof = 1;
			continue;
		}
		ateof = 0;
		if (length != -1)
			length -= n;
		if (!(need -= n)) {
			/*
			 * XXX bcmp() is not quite right in the presence
			 * of multibyte characters.
			 */
			if (vflag == ALL || vflag == FIRST ||
			    valid_save == 0 ||
			    bcmp(curp, savp, blocksize) != 0) {
				if (vflag == DUP || vflag == FIRST)
					vflag = WAIT;
				return(curp);
			}
			if (vflag == WAIT)
				(void)printf("*\n");
			vflag = DUP;
			address += blocksize;
			need = blocksize;
			nread = 0;
		}
		else
			nread += n;
	}
}

size_t
peek(u_char *buf, size_t nbytes)
{
	size_t n, nread;
	int c;

	if (length != -1 && nbytes > (unsigned int)length)
		nbytes = length;
	nread = 0;
	while (nread < nbytes && (c = getchar()) != EOF) {
		*buf++ = c;
		nread++;
	}
	n = nread;
	while (n-- > 0) {
		c = *--buf;
		ungetc(c, stdin);
	}
	return (nread);
}

int
next(char **argv)
{
	static int done;
	int statok;

	if (argv) {
		_argv = argv;
		return(1);
	}
	for (;;) {
		if (*_argv) {
			done = 1;
			if (!(freopen(*_argv, "r", stdin))) {
				warn("%s", *_argv);
				exitval = 1;
				++_argv;
				continue;
			}
			statok = 1;
		} else {
			if (done++)
				return(0);
			statok = 0;
		}

#ifndef __APPLE__
		if (caph_limit_stream(fileno(stdin), CAPH_READ) < 0)
			err(1, "unable to restrict %s",
			    statok ? *_argv : "stdin");

		/*
		 * We've opened our last input file; enter capsicum sandbox.
		 */
		if (statok == 0 || *(_argv + 1) == NULL) {
			if (caph_enter() < 0)
				err(1, "unable to enter capability mode");
		}
#endif

		if (skip)
			doskip(statok ? *_argv : "stdin", statok);
		if (*_argv)
			++_argv;
		if (!skip)
			return(1);
	}
	/* NOTREACHED */
}

void
doskip(const char *fname, int statok)
{
	int type;
	struct stat sb;

	if (statok) {
		if (fstat(fileno(stdin), &sb))
			err(1, "%s", fname);
		if (S_ISREG(sb.st_mode) && skip > sb.st_size) {
			address += sb.st_size;
			skip -= sb.st_size;
			return;
		}
	}
	if (!statok || S_ISFIFO(sb.st_mode) || S_ISSOCK(sb.st_mode)) {
		noseek();
		return;
	}
	if (S_ISCHR(sb.st_mode) || S_ISBLK(sb.st_mode)) {
		if (ioctl(fileno(stdin), FIODTYPE, &type))
			err(1, "%s", fname);
		/*
		 * Most tape drives don't support seeking,
		 * yet fseek() would succeed.
		 */
		if (type & D_TAPE) {
			noseek();
			return;
		}
	}
	if (fseeko(stdin, skip, SEEK_SET)) {
		noseek();
		return;
	}
	address += skip;
	skip = 0;
}

static void
noseek(void)
{
	int count;
	for (count = 0; count < skip; ++count)
		if (getchar() == EOF)
			break;
	address += count;
	skip -= count;
}
