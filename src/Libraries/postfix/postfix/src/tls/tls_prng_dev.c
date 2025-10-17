/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 4, 2022.
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
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include <errno.h>

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

/* tls_prng_dev_open - open entropy device */

TLS_PRNG_SRC *tls_prng_dev_open(const char *name, int timeout)
{
    const char *myname = "tls_prng_dev_open";
    TLS_PRNG_SRC *dev;
    int     fd;

    if ((fd = open(name, O_RDONLY, 0)) < 0) {
	if (msg_verbose)
	    msg_info("%s: cannot open entropy device %s: %m", myname, name);
	return (0);
    } else {
	dev = (TLS_PRNG_SRC *) mymalloc(sizeof(*dev));
	dev->fd = fd;
	dev->name = mystrdup(name);
	dev->timeout = timeout;
	if (msg_verbose)
	    msg_info("%s: opened entropy device %s", myname, name);
	return (dev);
    }
}

/* tls_prng_dev_read - update internal PRNG from device */

ssize_t tls_prng_dev_read(TLS_PRNG_SRC *dev, size_t len)
{
    const char *myname = "tls_prng_dev_read";
    unsigned char buffer[UCHAR_MAX];
    ssize_t count;
    size_t  rand_bytes;

    if (len <= 0)
	msg_panic("%s: bad read length: %ld", myname, (long) len);

    if (len > sizeof(buffer))
	rand_bytes = sizeof(buffer);
    else
	rand_bytes = len;
    errno = 0;
    count = timed_read(dev->fd, buffer, rand_bytes, dev->timeout, (void *) 0);
    if (count > 0) {
	if (msg_verbose)
	    msg_info("%s: read %ld bytes from entropy device %s",
		     myname, (long) count, dev->name);
	RAND_seed(buffer, count);
    } else {
	if (msg_verbose)
	    msg_info("%s: cannot read %ld bytes from entropy device %s: %m",
		     myname, (long) rand_bytes, dev->name);
    }
    return (count);
}

/* tls_prng_dev_close - disconnect from EGD server */

int     tls_prng_dev_close(TLS_PRNG_SRC *dev)
{
    const char *myname = "tls_prng_dev_close";
    int     err;

    if (msg_verbose)
	msg_info("%s: close entropy device %s", myname, dev->name);
    err = close(dev->fd);
    myfree(dev->name);
    myfree((void *) dev);
    return (err);
}

#endif
