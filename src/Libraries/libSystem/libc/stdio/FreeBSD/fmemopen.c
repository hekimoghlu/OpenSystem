/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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

#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "local.h"

struct fmemopen_cookie
{
	char	*buf;	/* pointer to the memory region */
	bool	 own;	/* did we allocate the buffer ourselves? */
	char     bin;   /* is this a binary buffer? */
	size_t	 size;	/* buffer length in bytes */
	size_t	 len;	/* data length in bytes */
	size_t	 off;	/* current offset into the buffer */
};

static int	fmemopen_read(void *cookie, char *buf, int nbytes);
static int	fmemopen_write(void *cookie, const char *buf, int nbytes);
static fpos_t	fmemopen_seek(void *cookie, fpos_t offset, int whence);
static int	fmemopen_close(void *cookie);

FILE *
fmemopen(void * __restrict buf, size_t size, const char * __restrict mode)
{
	struct fmemopen_cookie *ck;
	FILE *f;
	int flags, rc;

	/*
	 * POSIX says we shall return EINVAL if size is 0.
	 */
	if (size == 0) {
		errno = EINVAL;
		return (NULL);
	}

	/*
	 * Retrieve the flags as used by open(2) from the mode argument, and
	 * validate them.
	 */
	rc = __sflags(mode, &flags);
	if (rc == 0) {
		errno = EINVAL;
		return (NULL);
	}

	/*
	 * There's no point in requiring an automatically allocated buffer
	 * in write-only mode.
	 */
	if (!(flags & O_RDWR) && buf == NULL) {
		errno = EINVAL;
		return (NULL);
	}
	
	ck = malloc(sizeof(struct fmemopen_cookie));
	if (ck == NULL) {
		return (NULL);
	}

	ck->off  = 0;
	ck->size = size;

	/* Check whether we have to allocate the buffer ourselves. */
	ck->own = ((ck->buf = buf) == NULL);
	if (ck->own) {
		ck->buf = malloc(size);
		if (ck->buf == NULL) {
			free(ck);
			return (NULL);
		}
	}

	/*
	 * POSIX distinguishes between w+ and r+, in that w+ is supposed to
	 * truncate the buffer.
	 */
	if (ck->own || mode[0] == 'w') {
		ck->buf[0] = '\0';
	}

	/* Check for binary mode. */
	ck->bin = strchr(mode, 'b') != NULL;

	/*
	 * The size of the current buffer contents is set depending on the
	 * mode:
	 * 
	 * for append (text-mode), the position of the first NULL byte, or the
	 * size of the buffer if none is found
	 *
	 * for append (binary-mode), the size of the buffer
	 * 
	 * for read, the size of the buffer
	 * 
	 * for write, 0
	 */
	switch (mode[0]) {
	case 'a':
		ck->off = ck->len = strnlen(ck->buf, ck->size);
		break;
	case 'r':
		ck->len = size;
		break;
	case 'w':
		ck->len = 0;
		break;
	}

	f = funopen(ck,
	    flags & O_WRONLY ? NULL : fmemopen_read, 
	    flags & O_RDONLY ? NULL : fmemopen_write,
	    fmemopen_seek, fmemopen_close);

	if (f == NULL) {
		if (ck->own)
			free(ck->buf);
		free(ck);
		return (NULL);
	}

	if (mode[0] == 'a')
		f->_flags |= __SAPP;

	/*
	 * Turn off buffering, so a write past the end of the buffer
	 * correctly returns a short object count.
	 */
	setvbuf(f, NULL, _IONBF, 0);

	return (f);
}

static int
fmemopen_read(void *cookie, char *buf, int nbytes)
{
	struct fmemopen_cookie *ck = cookie;

	if (nbytes > ck->len - ck->off)
		nbytes = ck->len - ck->off;

	if (nbytes == 0)
		return (0);

	memcpy(buf, ck->buf + ck->off, nbytes);

	ck->off += nbytes;

	return (nbytes);
}

static int
fmemopen_write(void *cookie, const char *buf, int nbytes)
{
	struct fmemopen_cookie *ck = cookie;

	if (nbytes > ck->size - ck->off)
		nbytes = ck->size - ck->off;

	if (nbytes == 0)
		return (0);

	memcpy(ck->buf + ck->off, buf, nbytes);

	ck->off += nbytes;

	if (ck->off > ck->len)
		ck->len = ck->off;

	/*
	 * We append a NULL byte if all these conditions are met:
	 * - the buffer is not binary
	 * - the buffer is not full
	 * - the data just written doesn't already end with a NULL byte
	 */
	if (!ck->bin && ck->off < ck->size && ck->buf[ck->off - 1] != '\0')
		ck->buf[ck->off] = '\0';

	return (nbytes);
}

static fpos_t
fmemopen_seek(void *cookie, fpos_t offset, int whence)
{
	struct fmemopen_cookie *ck = cookie;


	switch (whence) {
	case SEEK_SET:
		if (offset > ck->size) {
			errno = EINVAL;
			return (-1);
		}
		ck->off = offset;
		break;

	case SEEK_CUR:
		if (ck->off + offset > ck->size) {
			errno = EINVAL;
			return (-1);
		}
		ck->off += offset;
		break;

	case SEEK_END:
		if (offset > 0 || -offset > ck->len) {
			errno = EINVAL;
			return (-1);
		}
		ck->off = ck->len + offset;
		break;

	default:
		errno = EINVAL;
		return (-1);
	}

	return (ck->off);
}

static int
fmemopen_close(void *cookie)
{
	struct fmemopen_cookie *ck = cookie;

	if (ck->own)
		free(ck->buf);

	free(ck);

	return (0);
}
