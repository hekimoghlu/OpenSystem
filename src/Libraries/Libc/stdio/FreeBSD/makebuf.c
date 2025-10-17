/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 20, 2023.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)makebuf.c	8.1 (Berkeley) 6/4/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include "namespace.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "un-namespace.h"

#include "libc_private.h"
#include "local.h"
#include <xlocale/_stdio.h>
#include <xlocale/_stdlib.h>
#include <os/once_private.h>

#ifdef FEATURE_SMALL_STDIOBUF
# define MAXBUFSIZE	(1 << 12)
#else
# define MAXBUFSIZE	(1 << 24)
#endif

#define TTYBUFSIZE	4096
#define MAXEVPSIZE 16

static char __fallback_evp[MAXEVPSIZE];
static char __stdin_evp[MAXEVPSIZE];
static char __stdout_evp[MAXEVPSIZE];
static char __stderr_evp[MAXEVPSIZE];

static void
__loadevp(const char *key, char destination[MAXEVPSIZE])
{
	char *evp = getenv(key);
	if (evp != NULL)
		strlcpy(destination, evp, MAXEVPSIZE);
}

static void
__evpinit(void __unused *unused)
{
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
 */__loadevp("STDBUF", __fallback_evp);
	__loadevp("STDBUF0", __stdin_evp);
	__loadevp("STDBUF1", __stdout_evp);
	__loadevp("STDBUF2", __stderr_evp);
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
 */__loadevp("_STDBUF_I", __stdin_evp);
	__loadevp("_STDBUF_O", __stdout_evp);
	__loadevp("_STDBUF_E", __stderr_evp);
}

static char *
__getevp(int fd)
{
	static os_once_t predicate;
	os_once(&predicate, NULL, __evpinit);

	switch (fd) {
	case STDIN_FILENO:
		if (__stdin_evp[0] != '\0')
			return __stdin_evp;
		break;
	case STDOUT_FILENO:
		if (__stdout_evp[0] != '\0')
			return __stdout_evp;
		break;
	case STDERR_FILENO:
		if (__stderr_evp[0] != '\0')
			return __stderr_evp;
		break;
	}
	if (__fallback_evp[0] != '\0')
		return __fallback_evp;
	return NULL;
}

/*
 * Internal routine to determine environment override buffering for a file.
 *
 * Sections of the below taken from NetBSD's version of this file under the same license.
 */
static int
__senvbuf(FILE *fp, size_t *bufsize, int *couldbetty)
{
	char *evp;
	int flags = 0; // default = fully buffered
	size_t size = 0;

	if ((evp = __getevp(fp->_file)) == NULL || *evp == '\0')
		return 0;
	/*
	 * NetBSD style: [UuLlFf] followed by an optional size
	 * GNU style: [0L] or a size
	 * FreeBSD style: [0LB] or a size
	 * Synthesis: optional [0UuLlFfB] followed by optional size.
	 */
	switch (*evp) {
	case '0':
	case 'U':
	case 'u':
		evp++;
		flags = __SNBF;
		break;
	case 'L':
	case 'l':
		evp++;
		flags = __SLBF;
		break;
	case 'F':
	case 'f':
	case 'B':
		evp++;
		break;
	}
	if (flags == __SNBF && *evp != '\0')
		return 0;
	for (; isdigit((unsigned char)*evp); evp++)
		size = size * 10 + *evp - '0';
	/*
	 * GNU accepts suffixes up to Z and has different notations for
	 * binary and decimal.  FreeBSD accepts suffixes up to G.  We'll
	 * settle for [Kk] and M and not bother with the distinction
	 * between binary and decimal.
	 */
	switch (*evp) {
	case 'M':
		evp++;
		size *= 1024 * 1024;
		break;
	case 'K':
	case 'k':
		evp++;
		size *= 1024;
		break;
	case 'B':
	case '\0':
		break;
	default:
		return 0;
	}
	if (*evp == 'B')
		evp++;
	if (*evp != '\0')
		return 0;
	*couldbetty = 0;
	*bufsize = size > MAXBUFSIZE ? MAXBUFSIZE : size;
	return flags;
}

/*
 * Allocate a file buffer, or switch to unbuffered I/O.
 * Per the ANSI C standard, ALL tty devices default to line buffered.
 *
 * As a side effect, we set __SOPT or __SNPT (en/dis-able fseek
 * optimisation) right after the _fstat() that finds the buffer size.
 */
void
__smakebuf(FILE *fp)
{
	void *p;
	int flags;
	size_t size;
	int couldbetty;

	if (fp->_flags & __SNBF) {
		fp->_bf._base = fp->_p = fp->_nbuf;
		fp->_bf._size = 1;
		return;
	}
	flags = __swhatbuf(fp, &size, &couldbetty);
	if (fp->_file >= 0) {
		flags |= __senvbuf(fp, &size, &couldbetty);

		if (flags & __SNBF) {
			fp->_flags |= __SNBF;
			fp->_bf._base = fp->_p = fp->_nbuf;
			fp->_bf._size = 1;
			return;
		}
		if (size == 0)
			size = BUFSIZ;
	}

	if (couldbetty && isatty(fp->_file)) {
		flags |= __SLBF;
		/* st_blksize for ttys is 128K, so make it more reasonable */
		if (size > TTYBUFSIZE)
			fp->_blksize = size = TTYBUFSIZE;
	}
	if ((p = malloc(size)) == NULL) {
		fp->_flags |= __SNBF;
		fp->_bf._base = fp->_p = fp->_nbuf;
		fp->_bf._size = 1;
		return;
	}
#ifdef __APPLE__
	__cleanup = 1;
#else
	__cleanup = _cleanup;
#endif // __APPLE__
	flags |= __SMBF;
	fp->_bf._base = fp->_p = p;
	fp->_bf._size = size;
	fp->_flags |= flags;
}

/*
 * Internal routine to determine `proper' buffering for a file.
 */
int
__swhatbuf(FILE *fp, size_t *bufsize, int *couldbetty)
{
	struct stat st;

	if (fp->_file < 0 || _fstat(fp->_file, &st) < 0) {
		*couldbetty = 0;
		*bufsize = BUFSIZ;
		return (__SNPT);
	}

	/* could be a tty iff it is a character device */
	*couldbetty = (st.st_mode & S_IFMT) == S_IFCHR;
	if (st.st_blksize <= 0) {
		*bufsize = BUFSIZ;
		return (__SNPT);
	}

	/*
	 * Optimise fseek() only if it is a regular file.  (The test for
	 * __sseek is mainly paranoia.)  It is safe to set _blksize
	 * unconditionally; it will only be used if __SOPT is also set.
	 */
	fp->_blksize = *bufsize = st.st_blksize > MAXBUFSIZE ? MAXBUFSIZE : st.st_blksize;
	return ((st.st_mode & S_IFMT) == S_IFREG && fp->_seek == __sseek ?
	    __SOPT : __SNPT);
}

