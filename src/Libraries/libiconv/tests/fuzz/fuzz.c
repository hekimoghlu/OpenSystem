/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
#include <sys/param.h>

#include <assert.h>
#include <iconv.h>
#include <locale.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef GBK

#define CS1	"UTF-8"
#define	CS2	"GBK"

#elif defined(CJK)

#define	CS1	"JISX0201-KANA"
#define CS2	"SHIFT_JIS-2004"

#elif defined(UCS)

#define	CS1	"UCS-2"
#define	CS2	"UTF-32"

#elif defined(EUC)

#define	CS1	"EUC-KR"
#ifndef CS2
#define	CS2	"UTF-32"
#endif

#elif defined(WCHAR2OPT)

#define WANT_MB
#define	CS1	"WCHAR_T"
#define	CS2	"UTF-8"

#elif defined(WCHAR2UNOPT)

#define WANT_MB
#define	CS1	"WCHAR_T"
#define	CS2	"SHIFT_JISX0213"

#endif

#ifndef CS1
#undef CS2

#define	CS1	"UTF-8"
#define CS2	"UTF-32"
#endif

#define	MAX_PIECES	32

static void
run_conversion(iconv_t cd, int pieces, const uint8_t *data, size_t sz)
{
	size_t bufsz = sz * 6;
	char outbuf[bufsz + 1];
	char *inptr, *outptr;
	size_t csz, insz, outsz;
	int chunksz;

	chunksz = sz / pieces;
	inptr = (char *)data;
	insz = sz;
	outptr = &outbuf[0];
	outsz = bufsz;
	for (int i = 0; i < pieces; i++) {
		csz = insz = MIN(chunksz, sz);
		(void)iconv(cd, &inptr, &insz, &outptr, &outsz);
		sz -= (csz - insz);
	}
}

int
LLVMFuzzerTestOneInput(const uint8_t *data, size_t sz)
{
	iconv_t cd;
	int pieces;

#ifdef WANT_MB
	static bool locale_set;

	if (!locale_set) {
		locale_set = true;
//		setlocale(LC_ALL, "en_US.UTF-8");
		setlocale(LC_ALL, "en_US.ISO8859-1");
	}
#endif	/* WANT_MB */


	if (sz == 0 || sz < MB_LEN_MAX)
		return (-1);

	/*
	 * We want predictable random, not actual random, so that we remain
	 * reproducible.
	 */
	srand(sz);
	/* Modulo bias doesn't really matter for this. */
	pieces = (rand() % MIN(MAX_PIECES, sz / MB_LEN_MAX)) + 1;
#if 0
	for (size_t i = 0; i < sz; i++) {
		fprintf(stderr, "\\x%.02x", data[i]);
	}
	fprintf(stderr, "\n");
#endif
	cd = iconv_open(CS1, CS2);
	assert(cd != (iconv_t)-1);
	run_conversion(cd, pieces, data, sz);
	iconv_close(cd);

	cd = iconv_open(CS2, CS1);
	assert(cd != (iconv_t)-1);
	run_conversion(cd, pieces, data, sz);
	iconv_close(cd);

	return (0);
}

