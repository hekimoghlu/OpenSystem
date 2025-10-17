/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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
#include <time.h>

/* Utility library. */

#include <msg.h>
#include <vstring.h>
#include <events.h>

/* SLMs. */

#define STR(x) vstring_str(x)

/* msg_rate_delay - rate-limit message logging */

void    msg_rate_delay(time_t *stamp, int delay,
		               void (*log_fn) (const char *,...),
		               const char *fmt,...)
{
    const char *myname = "msg_rate_delay";
    static time_t saved_event_time;
    time_t  now;
    VSTRING *buf;
    va_list ap;

    /*
     * Sanity check.
     */
    if (delay < 0)
	msg_panic("%s: bad message rate delay: %d", myname, delay);

    /*
     * This function may be called frequently. Avoid an unnecessary syscall
     * if possible. Deal with the possibility that a program does not use the
     * events(3) engine, so that event_time() always produces the same
     * result.
     */
    if (msg_verbose == 0 && delay > 0) {
	if (saved_event_time == 0)
	    now = saved_event_time = event_time();
	else if ((now = event_time()) == saved_event_time)
	    now = time((time_t *) 0);

	/*
	 * Don't log if time is too early.
	 */
	if (*stamp + delay > now)
	    return;
	*stamp = now;
    }

    /*
     * OK to log. This is a low-rate event, so we can afford some overhead.
     */
    buf = vstring_alloc(100);
    va_start(ap, fmt);
    vstring_vsprintf(buf, fmt, ap);
    va_end(ap);
    log_fn("%s", STR(buf));
    vstring_free(buf);
}

#ifdef TEST

 /*
  * Proof-of-concept test program: log messages but skip messages during a
  * two-second gap.
  */
#include <unistd.h>

int     main(int argc, char **argv)
{
    int     n;
    time_t  stamp = 0;

    for (n = 0; n < 6; n++) {
	msg_rate_delay(&stamp, 2, msg_info, "text here %d", n);
	sleep(1);
    }
    return (0);
}

#endif
