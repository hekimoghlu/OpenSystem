/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 12, 2023.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char sccsid[] = "@(#)rand.c	8.1 (Berkeley) 6/14/93";
#endif /* LIBC_SCCS and not lint */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include "namespace.h"
#include <sys/time.h>          /* for sranddev() */
#include <sys/types.h>
#include <fcntl.h>             /* for sranddev() */
#include <stdlib.h>
#include <unistd.h>            /* for sranddev() */
#include "un-namespace.h"

#ifdef TEST
#include <stdio.h>
#endif /* TEST */

static int
do_rand(unsigned long *ctx)
{
#ifdef  USE_WEAK_SEEDING
/*
 * Historic implementation compatibility.
 * The random sequences do not vary much with the seed,
 * even with overflowing.
 */
	return ((*ctx = *ctx * 1103515245 + 12345) % ((u_long)RAND_MAX + 1));
#else   /* !USE_WEAK_SEEDING */
/*
 * Compute x = (7^5 * x) mod (2^31 - 1)
 * without overflowing 31 bits:
 *      (2^31 - 1) = 127773 * (7^5) + 2836
 * From "Random number generators: good ones are hard to find",
 * Park and Miller, Communications of the ACM, vol. 31, no. 10,
 * October 1988, p. 1195.
 */
	long hi, lo, x;

	/* Can't be initialized with 0, so use another value. */
	if (*ctx == 0)
		*ctx = 123459876;
	hi = *ctx / 127773;
	lo = *ctx % 127773;
	x = 16807 * lo - 2836 * hi;
	if (x < 0)
		x += 0x7fffffff;
	return ((*ctx = x) % ((u_long)RAND_MAX + 1));
#endif  /* !USE_WEAK_SEEDING */
}


int
rand_r(unsigned int *ctx)
{
	u_long val = (u_long) *ctx;
	int r = do_rand(&val);

	*ctx = (unsigned int) val;
	return (r);
}


static u_long next = 1;

int
rand()
{
	return (do_rand(&next));
}

void
srand(seed)
u_int seed;
{
	next = seed;
}


/*
 * sranddev:
 *
 * Many programs choose the seed value in a totally predictable manner.
 * This often causes problems.  We seed the generator using the much more
 * secure random(4) interface.
 */
void
sranddev(void)
{
	int fd, done;

	done = 0;
	fd = _open("/dev/random", O_RDONLY | O_CLOEXEC, 0);
	if (fd >= 0) {
		if (_read(fd, (void *) &next, sizeof(next)) == sizeof(next))
			done = 1;
		_close(fd);
	}

	if (!done) {
		struct timeval tv;

		gettimeofday(&tv, NULL);
		srand((getpid() << 16) ^ tv.tv_sec ^ tv.tv_usec);
	}
}


#ifdef TEST

main()
{
    int i;
    unsigned myseed;

    printf("seeding rand with 0x19610910: \n");
    srand(0x19610910);

    printf("generating three pseudo-random numbers:\n");
    for (i = 0; i < 3; i++)
    {
	printf("next random number = %d\n", rand());
    }

    printf("generating the same sequence with rand_r:\n");
    myseed = 0x19610910;
    for (i = 0; i < 3; i++)
    {
	printf("next random number = %d\n", rand_r(&myseed));
    }

    return 0;
}

#endif /* TEST */

