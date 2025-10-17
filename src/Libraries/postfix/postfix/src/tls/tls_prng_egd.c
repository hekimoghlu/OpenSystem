/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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
#include <unistd.h>
#include <limits.h>

#ifndef UCHAR_MAX
#define UCHAR_MAX 0xff
#endif

/* OpenSSL library. */

#ifdef USE_TLS
#include <openssl/rand.h>		/* For the PRNG */

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <connect.h>
#include <iostuff.h>

/* TLS library. */

#include <tls_prng.h>

/* tls_prng_egd_open - connect to EGD server */

TLS_PRNG_SRC *tls_prng_egd_open(const char *name, int timeout)
{
    const char *myname = "tls_prng_egd_open";
    TLS_PRNG_SRC *egd;
    int     fd;

    if (msg_verbose)
	msg_info("%s: connect to EGD server %s", myname, name);

    if ((fd = unix_connect(name, BLOCKING, timeout)) < 0) {
	if (msg_verbose)
	    msg_info("%s: cannot connect to EGD server %s: %m", myname, name);
	return (0);
    } else {
	egd = (TLS_PRNG_SRC *) mymalloc(sizeof(*egd));
	egd->fd = fd;
	egd->name = mystrdup(name);
	egd->timeout = timeout;
	if (msg_verbose)
	    msg_info("%s: connected to EGD server %s", myname, name);
	return (egd);
    }
}

/* tls_prng_egd_read - update internal PRNG from EGD server */

ssize_t tls_prng_egd_read(TLS_PRNG_SRC *egd, size_t len)
{
    const char *myname = "tls_prng_egd_read";
    unsigned char buffer[UCHAR_MAX];
    ssize_t count;

    if (len <= 0)
	msg_panic("%s: bad length %ld", myname, (long) len);

    buffer[0] = 1;
    buffer[1] = (len > UCHAR_MAX ? UCHAR_MAX : len);

    if (timed_write(egd->fd, buffer, 2, egd->timeout, (void *) 0) != 2) {
	msg_info("cannot write to EGD server %s: %m", egd->name);
	return (-1);
    }
    if (timed_read(egd->fd, buffer, 1, egd->timeout, (void *) 0) != 1) {
	msg_info("cannot read from EGD server %s: %m", egd->name);
	return (-1);
    }
    count = buffer[0];
    if (count > sizeof(buffer))
	count = sizeof(buffer);
    if (count == 0) {
	msg_info("EGD server %s reports zero bytes available", egd->name);
	return (-1);
    }
    if (timed_read(egd->fd, buffer, count, egd->timeout, (void *) 0) != count) {
	msg_info("cannot read %ld bytes from EGD server %s: %m",
		 (long) count, egd->name);
	return (-1);
    }
    if (msg_verbose)
	msg_info("%s: got %ld bytes from EGD server %s", myname,
		 (long) count, egd->name);
    RAND_seed(buffer, count);
    return (count);
}

/* tls_prng_egd_close - disconnect from EGD server */

int     tls_prng_egd_close(TLS_PRNG_SRC *egd)
{
    const char *myname = "tls_prng_egd_close";
    int     err;

    if (msg_verbose)
	msg_info("%s: close EGD server %s", myname, egd->name);
    err = close(egd->fd);
    myfree(egd->name);
    myfree((void *) egd);
    return (err);
}

#endif
