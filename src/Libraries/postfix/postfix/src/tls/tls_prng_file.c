/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 20, 2024.
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

/* tls_prng_file_open - open entropy file */

TLS_PRNG_SRC *tls_prng_file_open(const char *name, int timeout)
{
    const char *myname = "tls_prng_file_open";
    TLS_PRNG_SRC *fh;
    int     fd;

    if ((fd = open(name, O_RDONLY, 0)) < 0) {
	if (msg_verbose)
	    msg_info("%s: cannot open entropy file %s: %m", myname, name);
	return (0);
    } else {
	fh = (TLS_PRNG_SRC *) mymalloc(sizeof(*fh));
	fh->fd = fd;
	fh->name = mystrdup(name);
	fh->timeout = timeout;
	if (msg_verbose)
	    msg_info("%s: opened entropy file %s", myname, name);
	return (fh);
    }
}

/* tls_prng_file_read - update internal PRNG from entropy file */

ssize_t tls_prng_file_read(TLS_PRNG_SRC *fh, size_t len)
{
    const char *myname = "tls_prng_file_read";
    char    buffer[8192];
    ssize_t to_read;
    ssize_t count;

    if (msg_verbose)
	msg_info("%s: seed internal pool from file %s", myname, fh->name);

    if (lseek(fh->fd, 0, SEEK_SET) < 0) {
	if (msg_verbose)
	    msg_info("cannot seek entropy file %s: %m", fh->name);
	return (-1);
    }
    errno = 0;
    for (to_read = len; to_read > 0; to_read -= count) {
	if ((count = timed_read(fh->fd, buffer, to_read > sizeof(buffer) ?
				sizeof(buffer) : to_read,
				fh->timeout, (void *) 0)) < 0) {
	    if (msg_verbose)
		msg_info("cannot read entropy file %s: %m", fh->name);
	    return (-1);
	}
	if (count == 0)
	    break;
	RAND_seed(buffer, count);
    }
    if (msg_verbose)
	msg_info("read %ld bytes from entropy file %s: %m",
		 (long) (len - to_read), fh->name);
    return (len - to_read);
}

/* tls_prng_file_close - close entropy file */

int     tls_prng_file_close(TLS_PRNG_SRC *fh)
{
    const char *myname = "tls_prng_file_close";
    int     err;

    if (msg_verbose)
	msg_info("%s: close entropy file %s", myname, fh->name);
    err = close(fh->fd);
    myfree(fh->name);
    myfree((void *) fh);
    return (err);
}

#endif
