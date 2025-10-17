/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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

#include <sys/endian.h>
#include <sys/types.h>

#include <err.h>
#include <errno.h>
#include <iconv.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * iconv_open must return (iconv_t)-1 on non-existing encoding
 * and set errno to EINVAL.
 */
static int
open_1(void)
{
	iconv_t cd;

	errno = 0;
	cd = iconv_open("nonexisting", "foobar");

	if ((cd == (iconv_t)-1) && (errno == EINVAL))
		return (0);
	else {
		iconv_close(cd);
		return (1);
	}
}

/*
 * iconv_open must return (iconv_t)-1 if too much files are open
 * and set errno to ENFILE.
 */
#define	MAX_LIMIT	1025
static int
open_2(void)
{
	iconv_t cd[MAX_LIMIT];
	int i, ret;

	errno = 0;
	for (i = 0; i < MAX_LIMIT; i++) {
		cd[i] = iconv_open("ASCII", "UTF8");
		if (cd[i] == (iconv_t)-1)
			break;
	}

	ret = (cd[i] == (iconv_t)-1) && ((errno == ENFILE) ||
	    (errno == EMFILE))  ? 0 : 1;
	for (; i > 0; i--)
		iconv_close(cd[i]);
	return (ret);
}

/*
 * iconv_close must return (iconv_t)-1 if conversion descriptor is
 * invalid and set errno to EBADF.
 */
static int
close_1(void)
{
	iconv_t cd = (iconv_t)-1;

	return ((iconv_close(cd) == -1) && (errno = EBADF) ? 0 : 1);
}

static int
conv_ebadf(void)
{
	iconv_t	cd = (iconv_t)-1;

	errno = 0;
	return ((iconv(cd, NULL, 0, NULL, 0) == (size_t)-1 && errno == EBADF) ? 0 : 1);
}

static int
conv_ret(void)
{
	iconv_t cd;
	size_t inbytesleft, outbytesleft;
	char *inptr;
	char *outptr;
	uint32_t outbuf[4];
	uint32_t inbuf[2] = { 0x00000151, 0x00000171 };

	if ((cd = iconv_open("ASCII", "UTF-32LE")) == (iconv_t)-1)
		return (1);

	inptr = (char *)inbuf;
	outptr = (char *)outbuf;
	inbytesleft = 8;
	outbytesleft = 16;

	return (iconv(cd, &inptr, &inbytesleft, &outptr, &outbytesleft) == 2 ? 0 : 1);
}

static int
conv_2big(void)
{
	iconv_t cd;
	size_t inbytesleft, outbytesleft;
	char *inptr;
	char *outptr;
	uint32_t inbuf[4];
	uint32_t outbuf[2];
	int ret;

	if ((cd = iconv_open("ASCII", "ASCII")) == (iconv_t)-1)
		return (1);

	inptr = (char *)inbuf;
	outptr = (char *)outbuf;
	inbytesleft = 16;
	outbytesleft = 8;

	errno = 0;
	ret = iconv(cd, &inptr, &inbytesleft, &outptr, &outbytesleft);

#ifdef VERBOSE
	printf("inptr - inbuf = %d\n", (const uint8_t *)inptr - (uint8_t *)inbuf);
	printf("inbytesleft = %d\n", inbytesleft);
	printf("outbytesleft = %d\n", outbytesleft);
	printf("outptr - outbuf = %d\n", (uint8_t *)outptr - (uint8_t *)outbuf);
	printf("errno = %d\n", errno);
	printf("ret = %d\n", (int)ret);
#endif

	if (((const uint8_t *)inptr - (uint8_t *)inbuf == 8) && (inbytesleft == 8)  &&
	    (outbytesleft == 0) && ((uint8_t *)outptr - (uint8_t *)outbuf == 8) &&
	    (errno == E2BIG) && ((size_t)ret == (size_t)-1))
		return (0);
	else
		return (1);
}

static int
conv_einval(void)
{
	iconv_t	 cd;
	size_t inbytesleft, outbytesleft;
	char *inptr;
	char *outptr;
	uint32_t outbuf[4];
        uint16_t inbuf[1] = { 0xEA42 };
	int ret;

	if ((cd = iconv_open("UTF-32", "BIG5")) == (iconv_t)-1)
		return (1);

	inptr = (char *)inbuf;
	outptr = (char *)outbuf;
	inbytesleft = 2;
	outbytesleft = 16;

	errno = 0;
	ret = iconv(cd, &inptr, &inbytesleft, &outptr, &outbytesleft);

#ifdef VERBOSE
	printf("inptr - inbuf = %d\n", (const uint8_t *)inptr - (uint8_t *)inbuf);
	printf("inbytesleft = %d\n", inbytesleft);
	printf("outbytesleft = %d\n", outbytesleft);
	printf("outptr - outbuf = %d\n", (uint8_t *)outptr - (uint8_t *)outbuf);
	printf("errno = %d\n", errno);
	printf("ret = %d\n", (int)ret);
#endif

	if (((const uint8_t *)inptr - (uint8_t *)inbuf == 1) && (inbytesleft == 1)  &&
	    (outbytesleft == 8) && ((uint8_t *)outptr - (uint8_t *)outbuf == 8) &&
	    (errno == EINVAL) && ((size_t)ret == (size_t)-1))
		return (0);
	else
		return (1);
}

static int
conv_eilseq(void)
{
	iconv_t cd;
	size_t inbytesleft, outbytesleft;
	char *inptr;
	char *outptr;
	uint32_t outbuf[4];
	uint16_t inbuf[1] = { 0x8AC0 };
	int ret;

	if ((cd = iconv_open("Latin2", "UTF-16LE")) == (iconv_t)-1)
		return (1);

	inptr = (char *)inbuf;
	outptr = (char *)outbuf;
	inbytesleft = 4;
	outbytesleft = 16;

	errno = 0;
	ret = iconv(cd, &inptr, &inbytesleft, &outptr, &outbytesleft);

#ifdef VERBOSE
	printf("inptr - inbuf = %d\n", (const uint8_t *)inptr - (uint8_t *)inbuf);
	printf("inbytesleft = %d\n", inbytesleft);
	printf("outbytesleft = %d\n", outbytesleft);
	printf("outptr - outbuf = %d\n", (uint8_t *)outptr - (uint8_t *)outbuf);
	printf("errno = %d\n", errno);
	printf("ret = %d\n", (int)ret);
#endif

	if (((const uint8_t *)inptr - (uint8_t *)inbuf == 0) && (inbytesleft == 4)  &&
	    (outbytesleft == 16) && ((uint8_t *)outptr - (uint8_t *)outbuf == 0) &&
	    (errno == EILSEQ) && ((size_t)ret == (size_t)-1))
		return (0);
	else
		return (1);
}

static void
test(int (tester) (void), const char * label)
{
	int ret;

	if ((ret = tester()))
		printf("%s failed (%d)\n", label, ret);
	else
		printf("%s succeeded\n", label);
}

int
main(void)
{

	test(open_1, "open_1");
	test(open_2, "open_2");
	test(close_1, "close_1");
	test(conv_ret, "conv_ret");
	test(conv_ebadf, "conv_ebadf");
	test(conv_2big, "conv_2big");
	test(conv_einval, "conv_einval");
	test(conv_eilseq, "conv_eilseq");
}
