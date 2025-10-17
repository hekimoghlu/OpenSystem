/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 17, 2023.
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
#include <stdlib.h>
#include <limits.h>
#include <errno.h>
#include <ctype.h>

/* Utility library. */

#include <msg.h>
#include <vstring.h>
#include <mymalloc.h>

/* Global library. */

#include <safe_ultostr.h>

/* Application-specific. */

#define STR	vstring_str
#define END	vstring_end
#define SWAP(type, a, b) { type temp; temp = a; a = b; b = temp; }

static unsigned char safe_chars[] =
"0123456789BCDFGHJKLMNPQRSTVWXYZbcdfghjklmnpqrstvwxyz";

#define SAFE_MAX_BASE	(sizeof(safe_chars) - 1)
#define SAFE_MIN_BASE	(2)

/* safe_ultostr - convert unsigned long to safe alphanumerical string */

char   *safe_ultostr(VSTRING *buf, unsigned long ulval, int base,
		             int padlen, int padchar)
{
    const char *myname = "safe_ultostr";
    char   *start;
    char   *last;
    int     i;

    /*
     * Sanity check.
     */
    if (base < SAFE_MIN_BASE || base > SAFE_MAX_BASE)
	msg_panic("%s: bad base: %d", myname, base);

    /*
     * First accumulate the result, backwards.
     */
    VSTRING_RESET(buf);
    while (ulval != 0) {
	VSTRING_ADDCH(buf, safe_chars[ulval % base]);
	ulval /= base;
    }
    while (VSTRING_LEN(buf) < padlen)
	VSTRING_ADDCH(buf, padchar);
    VSTRING_TERMINATE(buf);

    /*
     * Then, reverse the result.
     */
    start = STR(buf);
    last = END(buf) - 1;
    for (i = 0; i < VSTRING_LEN(buf) / 2; i++)
	SWAP(int, start[i], last[-i]);
    return (STR(buf));
}

/* safe_strtoul - convert safe alphanumerical string to unsigned long */

unsigned long safe_strtoul(const char *start, char **end, int base)
{
    const char *myname = "safe_strtoul";
    static unsigned char *char_map = 0;
    unsigned char *cp;
    unsigned long sum;
    unsigned long div_limit;
    unsigned long mod_limit;
    int     char_val;

    /*
     * Sanity check.
     */
    if (base < SAFE_MIN_BASE || base > SAFE_MAX_BASE)
	msg_panic("%s: bad base: %d", myname, base);

    /*
     * One-time initialization. Assume 8-bit bytes.
     */
    if (char_map == 0) {
	char_map = (unsigned char *) mymalloc(256);
	for (char_val = 0; char_val < 256; char_val++)
	    char_map[char_val] = SAFE_MAX_BASE;
	for (char_val = 0; char_val < SAFE_MAX_BASE; char_val++)
	    char_map[safe_chars[char_val]] = char_val;
    }

    /*
     * Per-call initialization.
     */
    sum = 0;
    div_limit = ULONG_MAX / base;
    mod_limit = ULONG_MAX % base;

    /*
     * Skip leading whitespace. We don't implement sign/base prefixes.
     */
    if (end)
	*end = (char *) start;
    while (ISSPACE(*start))
	++start;

    /*
     * Start the conversion.
     */
    errno = 0;
    for (cp = (unsigned char *) start; (char_val = char_map[*cp]) < base; cp++) {
	/* Return (ULONG_MAX, ERANGE) if the result is too large. */
	if (sum > div_limit
	    || (sum == div_limit && char_val > mod_limit)) {
	    sum = ULONG_MAX;
	    errno = ERANGE;
	    /* Skip "valid" characters, per the strtoul() spec. */
	    while (char_map[*++cp] < base)
		 /* void */ ;
	    break;
	}
	sum = sum * base + char_val;
    }
    /* Return (0, EINVAL) after no conversion. Test moved here 20131209. */
    if (cp == (unsigned char *) start)
	errno = EINVAL;
    else if (end)
	*end = (char *) cp;
    return (sum);
}

#ifdef TEST

 /*
  * Proof-of-concept test program. Read a number from stdin, convert to
  * string, and print the result.
  */
#include <stdio.h>			/* sscanf */
#include <vstream.h>
#include <vstring_vstream.h>

int     main(int unused_argc, char **unused_argv)
{
    VSTRING *buf = vstring_alloc(100);
    char   *junk;
    unsigned long ulval;
    int     base;
    char    ch;
    unsigned long ulval2;

#ifdef MISSING_STRTOUL
#define strtoul strtol
#endif

    /*
     * Hard-coded string-to-number test.
     */
    ulval2 = safe_strtoul("  ", &junk, 10);
    if (*junk == 0 || errno != EINVAL)
	msg_warn("input=' ' result=%lu errno=%m", ulval2);

    /*
     * Configurable number-to-string-to-number test.
     */
    while (vstring_get_nonl(buf, VSTREAM_IN) != VSTREAM_EOF) {
	ch = 0;
	if (sscanf(STR(buf), "%lu %d%c", &ulval, &base, &ch) != 2 || ch) {
	    msg_warn("bad input %s", STR(buf));
	} else {
	    (void) safe_ultostr(buf, ulval, base, 5, '0');
	    vstream_printf("%lu = %s\n", ulval, STR(buf));
	    ulval2 = safe_strtoul(STR(buf), &junk, base);
	    if (*junk || (ulval2 == ULONG_MAX && errno == ERANGE))
		msg_warn("%s: %m", STR(buf));
	    if (ulval2 != ulval)
		msg_warn("%lu != %lu", ulval2, ulval);
	}
	vstream_fflush(VSTREAM_OUT);
    }
    vstring_free(buf);
    return (0);
}

#endif
