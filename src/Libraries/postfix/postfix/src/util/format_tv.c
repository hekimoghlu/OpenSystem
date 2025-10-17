/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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
#include <sys_defs.h>

/* Utility library. */

#include <msg.h>
#include <format_tv.h>

/* Application-specific. */

#define MILLION	1000000

/* format_tv - print time with limited precision */

VSTRING *format_tv(VSTRING *buf, long sec, long usec,
		           int sig_dig, int max_dig)
{
    static int pow10[] = {1, 10, 100, 1000, 10000, 100000, 1000000};
    int     n;
    int     rem;
    int     wid;
    int     ures;

    /*
     * Sanity check.
     */
    if (max_dig < 0 || max_dig > 6)
	msg_panic("format_tv: bad maximum decimal count %d", max_dig);
    if (sec < 0 || usec < 0 || usec > MILLION)
	msg_panic("format_tv: bad time %lds %ldus", sec, usec);
    if (sig_dig < 1 || sig_dig > 6)
	msg_panic("format_tv: bad significant decimal count %d", sig_dig);
    ures = MILLION / pow10[max_dig];
    wid = pow10[sig_dig];

    /*
     * Adjust the resolution to suppress irrelevant digits.
     */
    if (ures < MILLION) {
	if (sec > 0) {
	    for (n = 1; sec >= n && n <= wid / 10; n *= 10)
		 /* void */ ;
	    ures = (MILLION / wid) * n;
	} else {
	    while (usec >= wid * ures)
		ures *= 10;
	}
    }

    /*
     * Round up the number if necessary. Leave thrash below the resolution.
     */
    if (ures > 1) {
	usec += ures / 2;
	if (usec >= MILLION) {
	    sec += 1;
	    usec -= MILLION;
	}
    }

    /*
     * Format the number. Truncate trailing null and thrash below resolution.
     */
    vstring_sprintf_append(buf, "%ld", sec);
    if (usec >= ures) {
	VSTRING_ADDCH(buf, '.');
	for (rem = usec, n = MILLION / 10; rem >= ures && n > 0; n /= 10) {
	    VSTRING_ADDCH(buf, "0123456789"[rem / n]);
	    rem %= n;
	}
    }
    VSTRING_TERMINATE(buf);
    return (buf);
}

#ifdef TEST

#include <stdio.h>
#include <stdlib.h>
#include <vstring_vstream.h>

int     main(int argc, char **argv)
{
    VSTRING *in = vstring_alloc(10);
    VSTRING *out = vstring_alloc(10);
    double  tval;
    int     sec;
    int     usec;
    int     sig_dig;
    int     max_dig;

    while (vstring_get_nonl(in, VSTREAM_IN) > 0) {
	vstream_printf(">> %s\n", vstring_str(in));
	if (vstring_str(in)[0] == 0 || vstring_str(in)[0] == '#')
	    continue;
	if (sscanf(vstring_str(in), "%lf %d %d", &tval, &sig_dig, &max_dig) != 3)
	    msg_fatal("bad input: %s", vstring_str(in));
	sec = (int) tval;			/* raw seconds */
	usec = (tval - sec) * MILLION;		/* raw microseconds */
	VSTRING_RESET(out);
	format_tv(out, sec, usec, sig_dig, max_dig);
	vstream_printf("%s\n", vstring_str(out));
	vstream_fflush(VSTREAM_OUT);
    }
    vstring_free(in);
    vstring_free(out);
    return (0);
}

#endif
