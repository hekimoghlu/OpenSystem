/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 21, 2025.
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

#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef __APPLE__
#include <libcasper.h>
#include <casper/cap_fileargs.h>
#endif

#include "extern.h"

/*
 * bytes -- read bytes to an offset from the end and display.
 *
 * This is the function that reads to a byte offset from the end of the input,
 * storing the data in a wrap-around buffer which is then displayed.  If the
 * rflag is set, the data is displayed in lines in reverse order, and this
 * routine has the usual nastiness of trying to find the newlines.  Otherwise,
 * it is displayed from the character closest to the beginning of the input to
 * the end.
 */
int
bytes(FILE *fp, const char *fn, off_t off)
{
	int ch, len, tlen;
	char *ep, *p, *t;
	int wrap;
	char *sp;

	if ((sp = p = malloc(off)) == NULL)
		err(1, "failed to allocate memory");

	for (wrap = 0, ep = p + off; (ch = getc(fp)) != EOF;) {
		*p = ch;
		if (++p == ep) {
			wrap = 1;
			p = sp;
		}
	}
	if (ferror(fp)) {
		ierr(fn);
		free(sp);
		return 1;
	}

	if (rflag) {
		for (t = p - 1, len = 0; t >= sp; --t, ++len)
			if (*t == '\n' && len) {
				WR(t + 1, len);
				len = 0;
		}
		if (wrap) {
			tlen = len;
			for (t = ep - 1, len = 0; t >= p; --t, ++len)
				if (*t == '\n') {
					if (len) {
						WR(t + 1, len);
						len = 0;
					}
					if (tlen) {
						WR(sp, tlen);
						tlen = 0;
					}
				}
			if (len)
				WR(t + 1, len);
			if (tlen)
				WR(sp, tlen);
		}
	} else {
		if (wrap && (len = ep - p))
			WR(p, len);
		len = p - sp;
		if (len)
			WR(sp, len);
	}

	free(sp);
	return 0;
}

/*
 * lines -- read lines to an offset from the end and display.
 *
 * This is the function that reads to a line offset from the end of the input,
 * storing the data in an array of buffers which is then displayed.  If the
 * rflag is set, the data is displayed in lines in reverse order, and this
 * routine has the usual nastiness of trying to find the newlines.  Otherwise,
 * it is displayed from the line closest to the beginning of the input to
 * the end.
 */
int
lines(FILE *fp, const char *fn, off_t off)
{
	struct {
		int blen;
		u_int len;
		char *l;
	} *llines;
	int ch, rc;
	char *p, *sp;
	int blen, cnt, recno, wrap;

	if ((llines = calloc(off, sizeof(*llines))) == NULL)
		err(1, "failed to allocate memory");
	p = sp = NULL;
	blen = cnt = recno = wrap = 0;
	rc = 0;

	while ((ch = getc(fp)) != EOF) {
		if (++cnt > blen) {
			if ((sp = realloc(sp, blen += 1024)) == NULL)
				err(1, "failed to allocate memory");
			p = sp + cnt - 1;
		}
		*p++ = ch;
		if (ch == '\n') {
			if ((int)llines[recno].blen < cnt) {
				llines[recno].blen = cnt + 256;
				if ((llines[recno].l = realloc(llines[recno].l,
				    llines[recno].blen)) == NULL)
					err(1, "failed to allocate memory");
			}
			bcopy(sp, llines[recno].l, llines[recno].len = cnt);
			cnt = 0;
			p = sp;
			if (++recno == off) {
				wrap = 1;
				recno = 0;
			}
		}
	}
	if (ferror(fp)) {
		ierr(fn);
		rc = 1;
		goto done;
	}
	if (cnt) {
		llines[recno].l = sp;
		sp = NULL;
		llines[recno].len = cnt;
		if (++recno == off) {
			wrap = 1;
			recno = 0;
		}
	}

	if (rflag) {
		for (cnt = recno - 1; cnt >= 0; --cnt)
			WR(llines[cnt].l, llines[cnt].len);
		if (wrap)
			for (cnt = off - 1; cnt >= recno; --cnt)
				WR(llines[cnt].l, llines[cnt].len);
	} else {
		if (wrap)
			for (cnt = recno; cnt < off; ++cnt)
				WR(llines[cnt].l, llines[cnt].len);
		for (cnt = 0; cnt < recno; ++cnt)
			WR(llines[cnt].l, llines[cnt].len);
	}
done:
	for (cnt = 0; cnt < off; cnt++)
		free(llines[cnt].l);
	free(sp);
	free(llines);
	return (rc);
}
