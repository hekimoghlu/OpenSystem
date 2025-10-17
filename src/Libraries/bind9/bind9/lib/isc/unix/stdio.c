/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 29, 2023.
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
/* $Id$ */

#include <config.h>

#include <errno.h>
#include <unistd.h>

#include <isc/stdio.h>
#include <isc/stat.h>
#include <isc/util.h>

#include "errno2result.h"

isc_result_t
isc_stdio_open(const char *filename, const char *mode, FILE **fp) {
	FILE *f;

	f = fopen(filename, mode);
	if (f == NULL)
		return (isc__errno2result(errno));
	*fp = f;
	return (ISC_R_SUCCESS);
}

isc_result_t
isc_stdio_close(FILE *f) {
	int r;

	r = fclose(f);
	if (r == 0)
		return (ISC_R_SUCCESS);
	else
		return (isc__errno2result(errno));
}

isc_result_t
isc_stdio_seek(FILE *f, off_t offset, int whence) {
	int r;

#ifdef HAVE_FSEEKO
	r = fseeko(f, offset, whence);
#else
	r = fseek(f, offset, whence);
#endif
	if (r == 0)
		return (ISC_R_SUCCESS);
	else
		return (isc__errno2result(errno));
}

isc_result_t
isc_stdio_tell(FILE *f, off_t *offsetp) {
	off_t r;

	REQUIRE(offsetp != NULL);

#ifdef HAVE_FTELLO
	r = ftello(f);
#else
	r = ftell(f);
#endif
	if (r >= 0) {
		*offsetp = r;
		return (ISC_R_SUCCESS);
	} else
		return (isc__errno2result(errno));
}

isc_result_t
isc_stdio_read(void *ptr, size_t size, size_t nmemb, FILE *f, size_t *nret) {
	isc_result_t result = ISC_R_SUCCESS;
	size_t r;

	clearerr(f);
	r = fread(ptr, size, nmemb, f);
	if (r != nmemb) {
		if (feof(f))
			result = ISC_R_EOF;
		else
			result = isc__errno2result(errno);
	}
	if (nret != NULL)
		*nret = r;
	return (result);
}

isc_result_t
isc_stdio_write(const void *ptr, size_t size, size_t nmemb, FILE *f,
	       size_t *nret)
{
	isc_result_t result = ISC_R_SUCCESS;
	size_t r;

	clearerr(f);
	r = fwrite(ptr, size, nmemb, f);
	if (r != nmemb)
		result = isc__errno2result(errno);
	if (nret != NULL)
		*nret = r;
	return (result);
}

isc_result_t
isc_stdio_flush(FILE *f) {
	int r;

	r = fflush(f);
	if (r == 0)
		return (ISC_R_SUCCESS);
	else
		return (isc__errno2result(errno));
}

/*
 * OpenBSD has deprecated ENOTSUP in favor of EOPNOTSUPP.
 */
#if defined(EOPNOTSUPP) && !defined(ENOTSUP)
#define ENOTSUP EOPNOTSUPP
#endif

isc_result_t
isc_stdio_sync(FILE *f) {
	struct stat buf;
	int r;

	if (fstat(fileno(f), &buf) != 0)
		return (isc__errno2result(errno));

	/*
	 * Only call fsync() on regular files.
	 */
	if ((buf.st_mode & S_IFMT) != S_IFREG)
		return (ISC_R_SUCCESS);

	r = fsync(fileno(f));
	if (r == 0)
		return (ISC_R_SUCCESS);
	else
		return (isc__errno2result(errno));
}

