/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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
#include "db_config.h"

#include "db_int.h"

/*
 * printf --
 *
 * PUBLIC: #ifndef HAVE_PRINTF
 * PUBLIC: int printf __P((const char *, ...));
 * PUBLIC: #endif
 */
#ifndef HAVE_PRINTF
int
#ifdef STDC_HEADERS
printf(const char *fmt, ...)
#else
printf(fmt, va_alist)
	const char *fmt;
	va_dcl
#endif
{
	va_list ap;
	size_t len;
	char buf[2048];    /* !!!: END OF THE STACK DON'T TRUST SPRINTF. */

#ifdef STDC_HEADERS
	va_start(ap, fmt);
#else
	va_start(ap);
#endif
	len = (size_t)vsnprintf(buf, sizeof(buf), fmt, ap);
#ifdef HAVE_BREW
	/*
	 * The BREW vsnprintf function return count includes the trailing
	 * nul-termination character.
	 */
	if (len > 0 && len <= sizeof(buf) && buf[len - 1] == '\0')
		--len;
#endif

	va_end(ap);

	/*
	 * We implement printf/fprintf with fwrite, because Berkeley DB uses
	 * fwrite in other places.
	 */
	return (fwrite(
	    buf, sizeof(char), (size_t)len, stdout) == len ? (int)len: -1);
}
#endif /* HAVE_PRINTF */

/*
 * fprintf --
 *
 * PUBLIC: #ifndef HAVE_PRINTF
 * PUBLIC: int fprintf __P((FILE *, const char *, ...));
 * PUBLIC: #endif
 */
#ifndef HAVE_PRINTF
int
#ifdef STDC_HEADERS
fprintf(FILE *fp, const char *fmt, ...)
#else
fprintf(fp, fmt, va_alist)
	FILE *fp;
	const char *fmt;
	va_dcl
#endif
{
	va_list ap;
	size_t len;
	char buf[2048];    /* !!!: END OF THE STACK DON'T TRUST SPRINTF. */

#ifdef STDC_HEADERS
	va_start(ap, fmt);
#else
	va_start(ap);
#endif
	len = vsnprintf(buf, sizeof(buf), fmt, ap);
#ifdef HAVE_BREW
	/*
	 * The BREW vsnprintf function return count includes the trailing
	 * nul-termination character.
	 */
	if (len > 0 && len <= sizeof(buf) && buf[len - 1] == '\0')
		--len;
#endif

	va_end(ap);

	/*
	 * We implement printf/fprintf with fwrite, because Berkeley DB uses
	 * fwrite in other places.
	 */
	return (fwrite(
	    buf, sizeof(char), (size_t)len, fp) == len ? (int)len: -1);
}
#endif /* HAVE_PRINTF */

/*
 * vfprintf --
 *
 * PUBLIC: #ifndef HAVE_PRINTF
 * PUBLIC: int vfprintf __P((FILE *, const char *, va_list));
 * PUBLIC: #endif
 */
#ifndef HAVE_PRINTF
int
vfprintf(fp, fmt, ap)
	FILE *fp;
	const char *fmt;
	va_list ap;
{
	size_t len;
	char buf[2048];    /* !!!: END OF THE STACK DON'T TRUST SPRINTF. */

	len = vsnprintf(buf, sizeof(buf), fmt, ap);
#ifdef HAVE_BREW
	/*
	 * The BREW vsnprintf function return count includes the trailing
	 * nul-termination character.
	 */
	if (len > 0 && len <= sizeof(buf) && buf[len - 1] == '\0')
		--len;
#endif

	/*
	 * We implement printf/fprintf with fwrite, because Berkeley DB uses
	 * fwrite in other places.
	 */
	return (fwrite(
	    buf, sizeof(char), (size_t)len, fp) == len ? (int)len: -1);
}
#endif /* HAVE_PRINTF */
