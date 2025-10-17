/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <rand.h>
#include <heim_threads.h>

#include <roken.h>

#include "randi.h"

#ifdef __APPLE_TARGET_EMBEDDED__

/*
 * Unix /dev/random
 */

int
_hc_unix_device_fd(int flags, const char **fn)
{
    static const char *rnd_devices[] = {
	"/dev/urandom",
	"/dev/random",
	"/dev/srandom",
	"/dev/arandom",
	NULL
    };
    const char **p;

    for(p = rnd_devices; *p; p++) {
	int fd = open(*p, flags | O_NDELAY);
	if(fd >= 0) {
	    if (fn)
		*fn = *p;
	    rk_cloexec(fd);
	    return fd;
	}
    }
    return -1;
}

static void
unix_seed(const void *indata, int size)
{
}


static int
unix_bytes(unsigned char *outdata, int size)
{
    ssize_t count;
    int fd;

    if (size < 0)
	return 0;
    else if (size == 0)
	return 1;

    fd = _hc_unix_device_fd(O_RDONLY, NULL);
    if (fd < 0)
	return 0;

    while (size > 0) {
	count = read(fd, outdata, size);
	if (count < 0 && errno == EINTR)
	    continue;
	else if (count <= 0) {
	    close(fd);
	    return 0;
	}
	outdata += count;
	size -= count;
    }
    close(fd);

    return 1;
}

static void
unix_cleanup(void)
{
}

static void
unix_add(const void *indata, int size, double entropi)
{
    unix_seed(indata, size);
}

static int
unix_pseudorand(unsigned char *outdata, int size)
{
    return unix_bytes(outdata, size);
}

static int
unix_status(void)
{
    int fd;

    fd = _hc_unix_device_fd(O_RDONLY, NULL);
    if (fd < 0)
	return 0;
    close(fd);

    return 1;
}

const RAND_METHOD hc_rand_unix_method = {
    unix_seed,
    unix_bytes,
    unix_cleanup,
    unix_add,
    unix_pseudorand,
    unix_status
};

const RAND_METHOD *
RAND_unix_method(void)
{
    return &hc_rand_unix_method;
}

#endif
