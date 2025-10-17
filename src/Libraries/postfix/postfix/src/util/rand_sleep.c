/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 14, 2022.
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
#include <unistd.h>
#include <time.h>

/* Utility library. */

#include <msg.h>
#include <myrand.h>
#include <iostuff.h>

/* rand_sleep - block for random time */

void    rand_sleep(unsigned delay, unsigned variation)
{
    const char *myname = "rand_sleep";
    unsigned usec;

    /*
     * Sanity checks.
     */
    if (delay == 0)
	msg_panic("%s: bad delay %d", myname, delay);
    if (variation > delay)
	msg_panic("%s: bad variation %d", myname, variation);

    /*
     * Use the semi-crappy random number generator.
     */
    usec = (delay - variation / 2) + variation * (double) myrand() / RAND_MAX;
    doze(usec);
}

#ifdef TEST

#include <msg_vstream.h>

int     main(int argc, char **argv)
{
    int     delay;
    int     variation;

    msg_vstream_init(argv[0], VSTREAM_ERR);
    if (argc != 3)
	msg_fatal("usage: %s delay variation", argv[0]);
    if ((delay = atoi(argv[1])) <= 0)
	msg_fatal("bad delay: %s", argv[1]);
    if ((variation = atoi(argv[2])) < 0)
	msg_fatal("bad variation: %s", argv[2]);
    rand_sleep(delay * 1000000, variation * 1000000);
    exit(0);
}

#endif
