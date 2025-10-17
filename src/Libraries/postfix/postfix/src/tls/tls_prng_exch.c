/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 1, 2023.
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

/* OpenSSL library. */

#ifdef USE_TLS
#include <openssl/rand.h>		/* For the PRNG */

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <iostuff.h>
#include <myflock.h>

/* TLS library. */

#include <tls_prng.h>

/* Application specific. */

#define TLS_PRNG_EXCH_SIZE	1024	/* XXX Why not configurable? */

/* tls_prng_exch_open - open PRNG exchange file */

TLS_PRNG_SRC *tls_prng_exch_open(const char *name)
{
    const char *myname = "tls_prng_exch_open";
    TLS_PRNG_SRC *eh;
    int     fd;

    if ((fd = open(name, O_RDWR | O_CREAT, 0600)) < 0)
	msg_fatal("%s: cannot open PRNG exchange file %s: %m", myname, name);
    eh = (TLS_PRNG_SRC *) mymalloc(sizeof(*eh));
    eh->fd = fd;
    eh->name = mystrdup(name);
    eh->timeout = 0;
    if (msg_verbose)
	msg_info("%s: opened PRNG exchange file %s", myname, name);
    return (eh);
}

/* tls_prng_exch_update - update PRNG exchange file */

void    tls_prng_exch_update(TLS_PRNG_SRC *eh)
{
    unsigned char buffer[TLS_PRNG_EXCH_SIZE];
    ssize_t count;

    /*
     * Update the PRNG exchange file. Since other processes may have added
     * entropy, we use a read-stir-write cycle.
     */
    if (myflock(eh->fd, INTERNAL_LOCK, MYFLOCK_OP_EXCLUSIVE) != 0)
	msg_fatal("cannot lock PRNG exchange file %s: %m", eh->name);
    if (lseek(eh->fd, 0, SEEK_SET) < 0)
	msg_fatal("cannot seek PRNG exchange file %s: %m", eh->name);
    if ((count = read(eh->fd, buffer, sizeof(buffer))) < 0)
	msg_fatal("cannot read PRNG exchange file %s: %m", eh->name);

    if (count > 0)
	RAND_seed(buffer, count);
    RAND_bytes(buffer, sizeof(buffer));

    if (lseek(eh->fd, 0, SEEK_SET) < 0)
	msg_fatal("cannot seek PRNG exchange file %s: %m", eh->name);
    if (write(eh->fd, buffer, sizeof(buffer)) != sizeof(buffer))
	msg_fatal("cannot write PRNG exchange file %s: %m", eh->name);
    if (myflock(eh->fd, INTERNAL_LOCK, MYFLOCK_OP_NONE) != 0)
	msg_fatal("cannot unlock PRNG exchange file %s: %m", eh->name);
}

/* tls_prng_exch_close - close PRNG exchange file */

void    tls_prng_exch_close(TLS_PRNG_SRC *eh)
{
    const char *myname = "tls_prng_exch_close";

    if (close(eh->fd) < 0)
	msg_fatal("close PRNG exchange file %s: %m", eh->name);
    if (msg_verbose)
	msg_info("%s: closed PRNG exchange file %s", myname, eh->name);
    myfree(eh->name);
    myfree((void *) eh);
}

#endif
