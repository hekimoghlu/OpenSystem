/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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
#include <sys/time.h>			/* gettimeofday() */
#include <unistd.h>			/* getpid() */

#ifdef USE_TLS

/* OpenSSL library. */

#include <openssl/rand.h>		/* RAND_seed() */

/* Utility library. */

#include <msg.h>
#include <vstring.h>

/* TLS library. */

#include <tls_mgr.h>
#define TLS_INTERNAL
#include <tls.h>

/* Application-specific. */

/* tls_int_seed - add entropy to the pool by adding the time and PID */

void    tls_int_seed(void)
{
    static struct {
	pid_t   pid;
	struct timeval tv;
    }       randseed;

    if (randseed.pid == 0)
	randseed.pid = getpid();
    GETTIMEOFDAY(&randseed.tv);
    RAND_seed(&randseed, sizeof(randseed));
}

/* tls_ext_seed - request entropy from tlsmgr(8) server */

int     tls_ext_seed(int nbytes)
{
    VSTRING *buf;
    int     status;

    buf = vstring_alloc(nbytes);
    status = tls_mgr_seed(buf, nbytes);
    RAND_seed(vstring_str(buf), VSTRING_LEN(buf));
    vstring_free(buf);
    return (status == TLS_MGR_STAT_OK ? 0 : -1);
}

#endif
