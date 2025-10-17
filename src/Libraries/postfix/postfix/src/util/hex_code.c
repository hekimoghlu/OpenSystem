/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 9, 2022.
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
/* System library. */

#include <sys_defs.h>
#include <ctype.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <vstring.h>
#include <hex_code.h>

/* Application-specific. */

static const unsigned char hex_chars[] = "0123456789ABCDEF";

#define UCHAR_PTR(x) ((const unsigned char *)(x))

/* hex_encode - raw data to encoded */

VSTRING *hex_encode(VSTRING *result, const char *in, ssize_t len)
{
    const unsigned char *cp;
    int     ch;
    ssize_t count;

    VSTRING_RESET(result);
    for (cp = UCHAR_PTR(in), count = len; count > 0; count--, cp++) {
	ch = *cp;
	VSTRING_ADDCH(result, hex_chars[(ch >> 4) & 0xf]);
	VSTRING_ADDCH(result, hex_chars[ch & 0xf]);
    }
    VSTRING_TERMINATE(result);
    return (result);
}

/* hex_decode - encoded data to raw */

VSTRING *hex_decode(VSTRING *result, const char *in, ssize_t len)
{
    const unsigned char *cp;
    ssize_t count;
    unsigned int hex;
    unsigned int bin;

    VSTRING_RESET(result);
    for (cp = UCHAR_PTR(in), count = len; count > 0; cp += 2, count -= 2) {
	if (count < 2)
	    return (0);
	hex = cp[0];
	if (hex >= '0' && hex <= '9')
	    bin = (hex - '0') << 4;
	else if (hex >= 'A' && hex <= 'F')
	    bin = (hex - 'A' + 10) << 4;
	else if (hex >= 'a' && hex <= 'f')
	    bin = (hex - 'a' + 10) << 4;
	else
	    return (0);
	hex = cp[1];
	if (hex >= '0' && hex <= '9')
	    bin |= (hex - '0');
	else if (hex >= 'A' && hex <= 'F')
	    bin |= (hex - 'A' + 10);
	else if (hex >= 'a' && hex <= 'f')
	    bin |= (hex - 'a' + 10);
	else
	    return (0);
	VSTRING_ADDCH(result, bin);
    }
    VSTRING_TERMINATE(result);
    return (result);
}

#ifdef TEST

 /*
  * Proof-of-concept test program: convert to hexadecimal and back.
  */

#define STR(x)	vstring_str(x)
#define LEN(x)	VSTRING_LEN(x)

int     main(int unused_argc, char **unused_argv)
{
    VSTRING *b1 = vstring_alloc(1);
    VSTRING *b2 = vstring_alloc(1);
    char   *test = "this is a test";

#define DECODE(b,x,l) { \
	if (hex_decode((b),(x),(l)) == 0) \
	    msg_panic("bad hex: %s", (x)); \
    }
#define VERIFY(b,t) { \
	if (strcmp((b), (t)) != 0) \
	    msg_panic("bad test: %s", (b)); \
    }

    hex_encode(b1, test, strlen(test));
    DECODE(b2, STR(b1), LEN(b1));
    VERIFY(STR(b2), test);

    hex_encode(b1, test, strlen(test));
    hex_encode(b2, STR(b1), LEN(b1));
    hex_encode(b1, STR(b2), LEN(b2));
    DECODE(b2, STR(b1), LEN(b1));
    DECODE(b1, STR(b2), LEN(b2));
    DECODE(b2, STR(b1), LEN(b1));
    VERIFY(STR(b2), test);

    hex_encode(b1, test, strlen(test));
    hex_encode(b2, STR(b1), LEN(b1));
    hex_encode(b1, STR(b2), LEN(b2));
    hex_encode(b2, STR(b1), LEN(b1));
    hex_encode(b1, STR(b2), LEN(b2));
    DECODE(b2, STR(b1), LEN(b1));
    DECODE(b1, STR(b2), LEN(b2));
    DECODE(b2, STR(b1), LEN(b1));
    DECODE(b1, STR(b2), LEN(b2));
    DECODE(b2, STR(b1), LEN(b1));
    VERIFY(STR(b2), test);

    vstring_free(b1);
    vstring_free(b2);
    return (0);
}

#endif
