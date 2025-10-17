/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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

#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>

#include <krb5-types.h>

#include "roken.h"

#include <CommonCrypto/CommonCryptor.h>
#include "CCDGlue.h"

static int
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

int
CCRandomCopyBytes(CCRandomRef source, void *data, size_t size)
{
    uint8_t *outdata = data;
    ssize_t count;
    int fd;

    if (size == 0)
	return 0;

    fd = _hc_unix_device_fd(O_RDONLY, NULL);
    if (fd < 0)
	return EINVAL;

    while (size > 0) {
	count = read(fd, outdata, size);
	if (count < 0 && errno == EINTR)
	    continue;
	else if (count <= 0) {
	    close(fd);
	    return EINVAL;
	}
	outdata += count;
	size -= count;
    }
    close(fd);

    return 1;
}
