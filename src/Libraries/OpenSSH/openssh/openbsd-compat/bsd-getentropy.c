/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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
#include "includes.h"

#ifndef SSH_RANDOM_DEV
# define SSH_RANDOM_DEV "/dev/urandom"
#endif /* SSH_RANDOM_DEV */

#include <sys/types.h>
#ifdef HAVE_SYS_RANDOM_H
# include <sys/random.h>
#endif

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef WITH_OPENSSL
#include <openssl/rand.h>
#include <openssl/err.h>
#endif

#include "log.h"

int
_ssh_compat_getentropy(void *s, size_t len)
{
#if defined(WITH_OPENSSL) && defined(OPENSSL_PRNG_ONLY)
	if (RAND_bytes(s, len) <= 0)
		fatal("Couldn't obtain random bytes (error 0x%lx)",
		    (unsigned long)ERR_get_error());
#else
	int fd, save_errno;
	ssize_t r;
	size_t o = 0;

#ifdef WITH_OPENSSL
	if (RAND_bytes(s, len) == 1)
		return 0;
#endif
#ifdef HAVE_GETENTROPY
	if ((r = getentropy(s, len)) == 0)
		return 0;
#endif /* HAVE_GETENTROPY */
#ifdef HAVE_GETRANDOM
	if ((r = getrandom(s, len, 0)) > 0 && (size_t)r == len)
		return 0;
#endif /* HAVE_GETRANDOM */

	if ((fd = open(SSH_RANDOM_DEV, O_RDONLY)) == -1) {
		save_errno = errno;
		/* Try egd/prngd before giving up. */
		if (seed_from_prngd(s, len) == 0)
			return 0;
		fatal("Couldn't open %s: %s", SSH_RANDOM_DEV,
		    strerror(save_errno));
	}
	while (o < len) {
		r = read(fd, (u_char *)s + o, len - o);
		if (r < 0) {
			if (errno == EAGAIN || errno == EINTR ||
			    errno == EWOULDBLOCK)
				continue;
			fatal("read %s: %s", SSH_RANDOM_DEV, strerror(errno));
		}
		o += r;
	}
	close(fd);
#endif /* WITH_OPENSSL */
	return 0;
}
