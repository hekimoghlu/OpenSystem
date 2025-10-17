/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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
#include "archive_platform.h"

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_ZLIB_H
#include <zlib.h>
#endif
#ifdef HAVE_LZMA_H
#include <lzma.h>
#endif
#ifdef HAVE_BZLIB_H
#include <bzlib.h>
#endif
#ifdef HAVE_LZ4_H
#include <lz4.h>
#endif
#ifdef HAVE_ZSTD_H
#include <zstd.h>
#endif

#include "archive.h"
#include "archive_private.h"
#include "archive_string.h"

const char *
archive_version_details(void)
{
	static struct archive_string str;
	static int init = 0;
	const char *zlib = archive_zlib_version();
	const char *liblzma = archive_liblzma_version();
	const char *bzlib = archive_bzlib_version();
	const char *liblz4 = archive_liblz4_version();
	const char *libzstd = archive_libzstd_version();

	if (!init) {
		archive_string_init(&str);

		archive_strcat(&str, ARCHIVE_VERSION_STRING);
		if (zlib != NULL) {
			archive_strcat(&str, " zlib/");
			archive_strcat(&str, zlib);
		}
		if (liblzma) {
			archive_strcat(&str, " liblzma/");
			archive_strcat(&str, liblzma);
		}
		if (bzlib) {
			const char *p = bzlib;
			const char *sep = strchr(p, ',');
			if (sep == NULL)
				sep = p + strlen(p);
			archive_strcat(&str, " bz2lib/");
			archive_strncat(&str, p, sep - p);
		}
		if (liblz4) {
			archive_strcat(&str, " liblz4/");
			archive_strcat(&str, liblz4);
		}
		if (libzstd) {
			archive_strcat(&str, " libzstd/");
			archive_strcat(&str, libzstd);
		}
	}
	return str.s;
}

const char *
archive_zlib_version(void)
{
#ifdef HAVE_ZLIB_H
	return ZLIB_VERSION;
#else
	return NULL;
#endif
}

const char *
archive_liblzma_version(void)
{
#ifdef HAVE_LZMA_H
	return LZMA_VERSION_STRING;
#else
	return NULL;
#endif
}

const char *
archive_bzlib_version(void)
{
#ifdef HAVE_BZLIB_H
	return BZ2_bzlibVersion();
#else
	return NULL;
#endif
}

const char *
archive_liblz4_version(void)
{
#if defined(HAVE_LZ4_H) && defined(HAVE_LIBLZ4)
#define str(s) #s
#define NUMBER(x) str(x)
	return NUMBER(LZ4_VERSION_MAJOR) "." NUMBER(LZ4_VERSION_MINOR) "." NUMBER(LZ4_VERSION_RELEASE);
#undef NUMBER
#undef str
#else
	return NULL;
#endif
}

const char *
archive_libzstd_version(void)
{
#if HAVE_ZSTD_H && HAVE_LIBZSTD
	return ZSTD_VERSION_STRING;
#else
	return NULL;
#endif
}
