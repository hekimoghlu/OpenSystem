/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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

/* Utility library. */

#include <msg.h>

/* Application-specific. */

#include "sane_time.h"

/*
 * How many times shall we slow down the real clock when recovering from
 * time jump.
 */
#define SLEW_FACTOR 2

/* sane_time - get current time, protected against time warping */

time_t  sane_time(void)
{
    time_t  now;
    static time_t last_time, last_real;
    long    delta;
    static int fraction;
    static int warned;

    now = time((time_t *) 0);

    if ((delta = now - last_time) < 0 && last_time != 0) {
	if ((delta = now - last_real) < 0) {
	    msg_warn("%sbackward time jump detected -- slewing clock",
		     warned++ ? "another " : "");
	} else {
	    delta += fraction;
	    last_time += delta / SLEW_FACTOR;
	    fraction = delta % SLEW_FACTOR;
	}
    } else {
	if (warned) {
	    warned = 0;
	    msg_warn("backward time jump recovered -- back to normality");
	    fraction = 0;
	}
	last_time = now;
    }
    last_real = now;

    return (last_time);
}

#ifdef TEST

 /*
  * Proof-of-concept test program. Repeatedly print current system time and
  * time returned by sane_time(). Meanwhile, try stepping your system clock
  * back and forth to see what happens.
  */

#include <stdlib.h>
#include <msg_vstream.h>
#include <iostuff.h>			/* doze() */

int     main(int argc, char **argv)
{
    int     delay = 1000000;
    time_t  now;

    msg_vstream_init(argv[0], VSTREAM_ERR);

    if (argc == 2 && (delay = atol(argv[1]) * 1000) > 0)
	 /* void */ ;
    else if (argc != 1)
	msg_fatal("usage: %s [delay in ms (default 1 second)]", argv[0]);

    for (;;) {
	now = time((time_t *) 0);
	vstream_printf("real: %s", ctime(&now));
	now = sane_time();
	vstream_printf("fake: %s\n", ctime(&now));
	vstream_fflush(VSTREAM_OUT);
	doze(delay);
    }
}

#endif
