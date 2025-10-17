/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <assert.h>
#include <errno.h>
#include <limits.h>
#ifdef DEBUG
#include <stdint.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include <malloc_private.h>

/* XXX: There is no FPOS_MAX.  This assumes fpos_t is an off_t. */
#define	FPOS_MAX	OFF_MAX

struct wmemstream {
	wchar_t **bufp;
	size_t *sizep;
	ssize_t len;
	fpos_t offset;
	mbstate_t mbstate;
};

static int
wmemstream_grow(struct wmemstream *ms, fpos_t newoff)
{
	wchar_t *buf;
	ssize_t newsize;

	if (newoff < 0 || newoff >= SSIZE_MAX / sizeof(wchar_t))
		newsize = SSIZE_MAX / sizeof(wchar_t) - 1;
	else
		newsize = newoff;
	if (newsize > ms->len) {
		buf = reallocarray(*ms->bufp, newsize + 1, sizeof(wchar_t));
		if (buf != NULL) {
#ifdef DEBUG
			fprintf(stderr, "WMS: %p growing from %zd to %zd\n",
			    ms, ms->len, newsize);
#endif
			wmemset(buf + ms->len + 1, 0, newsize - ms->len);
			*ms->bufp = buf;
			ms->len = newsize;
			return (1);
		}
		return (0);
	}
	return (1);
}

static void
wmemstream_update(struct wmemstream *ms)
{

	assert(ms->len >= 0 && ms->offset >= 0);
	*ms->sizep = ms->len < ms->offset ? ms->len : ms->offset;
}

/*
 * Based on a starting multibyte state and an input buffer, determine
 * how many wchar_t's would be output.  This doesn't use mbsnrtowcs()
 * so that it can handle embedded null characters.
 */
static size_t
wbuflen(const mbstate_t *state, const char *buf, int len)
{
	mbstate_t lenstate;
	size_t charlen, count;

	count = 0;
	lenstate = *state;
	while (len > 0) {
		charlen = mbrlen(buf, len, &lenstate);
		if (charlen == (size_t)-1)
			return (-1);
		if (charlen == (size_t)-2)
			break;
		if (charlen == 0)
			/* XXX: Not sure how else to handle this. */
			charlen = 1;
		len -= charlen;
		buf += charlen;
		count++;
	}
	return (count);
}

static int
wmemstream_write(void *cookie, const char *buf, int len)
{
	struct wmemstream *ms;
	ssize_t consumed, wlen;
	size_t charlen;

	ms = cookie;
	wlen = wbuflen(&ms->mbstate, buf, len);
	if (wlen < 0) {
		errno = EILSEQ;
		return (-1);
	}
	if (!wmemstream_grow(ms, ms->offset + wlen))
		return (-1);

	/*
	 * This copies characters one at a time rather than using
	 * mbsnrtowcs() so it can properly handle embedded null
	 * characters.
	 */
	consumed = 0;
	while (len > 0 && ms->offset < ms->len) {
		charlen = mbrtowc(*ms->bufp + ms->offset, buf, len,
		    &ms->mbstate);
		if (charlen == (size_t)-1) {
			if (consumed == 0) {
				errno = EILSEQ;
				return (-1);
			}
			/* Treat it as a successful short write. */
			break;
		}
		if (charlen == 0)
			/* XXX: Not sure how else to handle this. */
			charlen = 1;
		if (charlen == (size_t)-2) {
			consumed += len;
			len = 0;
		} else {
			consumed += charlen;
			buf += charlen;
			len -= charlen;
			ms->offset++;
		}
	}
	wmemstream_update(ms);
#ifdef DEBUG
	fprintf(stderr, "WMS: write(%p, %d) = %zd\n", ms, len, consumed);
#endif
	return (consumed);
}

static fpos_t
wmemstream_seek(void *cookie, fpos_t pos, int whence)
{
	struct wmemstream *ms;
	fpos_t old;

	ms = cookie;
	old = ms->offset;
	switch (whence) {
	case SEEK_SET:
		/* _fseeko() checks for negative offsets. */
		assert(pos >= 0);
		ms->offset = pos;
		break;
	case SEEK_CUR:
		/* This is only called by _ftello(). */
		assert(pos == 0);
		break;
	case SEEK_END:
		if (pos < 0) {
			if (pos + ms->len < 0) {
#ifdef DEBUG
				fprintf(stderr,
				    "WMS: bad SEEK_END: pos %jd, len %zd\n",
				    (intmax_t)pos, ms->len);
#endif
				errno = EINVAL;
				return (-1);
			}
		} else {
			if (FPOS_MAX - ms->len < pos) {
#ifdef DEBUG
				fprintf(stderr,
				    "WMS: bad SEEK_END: pos %jd, len %zd\n",
				    (intmax_t)pos, ms->len);
#endif
				errno = EOVERFLOW;
				return (-1);
			}
		}
		ms->offset = ms->len + pos;
		break;
	}
	/* Reset the multibyte state if a seek changes the position. */
	if (ms->offset != old)
		memset(&ms->mbstate, 0, sizeof(ms->mbstate));
	wmemstream_update(ms);
#ifdef DEBUG
	fprintf(stderr, "WMS: seek(%p, %jd, %d) %jd -> %jd\n", ms,
	    (intmax_t)pos, whence, (intmax_t)old, (intmax_t)ms->offset);
#endif
	return (ms->offset);
}

static int
wmemstream_close(void *cookie)
{

	free(cookie);
	return (0);
}

FILE *
open_wmemstream(wchar_t **bufp, size_t *sizep)
{
	struct wmemstream *ms;
	int save_errno;
	FILE *fp;

	if (bufp == NULL || sizep == NULL) {
		errno = EINVAL;
		return (NULL);
	}
	*bufp = calloc(1, sizeof(wchar_t));
	if (*bufp == NULL)
		return (NULL);
	ms = malloc(sizeof(*ms));
	if (ms == NULL) {
		save_errno = errno;
		free(*bufp);
		*bufp = NULL;
		errno = save_errno;
		return (NULL);
	}
	ms->bufp = bufp;
	ms->sizep = sizep;
	ms->len = 0;
	ms->offset = 0;
	memset(&ms->mbstate, 0, sizeof(mbstate_t));
	wmemstream_update(ms);
	fp = funopen(ms, NULL, wmemstream_write, wmemstream_seek,
	    wmemstream_close);
	if (fp == NULL) {
		save_errno = errno;
		free(ms);
		free(*bufp);
		*bufp = NULL;
		errno = save_errno;
		return (NULL);
	}
	fwide(fp, 1);
	return (fp);
}
