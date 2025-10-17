/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 18, 2023.
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

#define RANDOM_SEED_SIZE 48

#ifdef WITH_OPENSSL

#include <sys/types.h>

#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <openssl/rand.h>
#include <openssl/crypto.h>
#include <openssl/err.h>

#include "openbsd-compat/openssl-compat.h"

#include "ssh.h"
#include "misc.h"
#include "xmalloc.h"
#include "atomicio.h"
#include "pathnames.h"
#include "log.h"
#include "sshbuf.h"
#include "ssherr.h"

/*
 * Portable OpenSSH PRNG seeding:
 * If OpenSSL has not "internally seeded" itself (e.g. pulled data from
 * /dev/random), then collect RANDOM_SEED_SIZE bytes of randomness from
 * PRNGd.
 */

void
seed_rng(void)
{
	unsigned char buf[RANDOM_SEED_SIZE];

	/* Initialise libcrypto */
	ssh_libcrypto_init();

	if (!ssh_compatible_openssl(OPENSSL_VERSION_NUMBER,
	    OpenSSL_version_num()))
		fatal("OpenSSL version mismatch. Built against %lx, you "
		    "have %lx", (u_long)OPENSSL_VERSION_NUMBER,
		    OpenSSL_version_num());

#ifndef OPENSSL_PRNG_ONLY
	if (RAND_status() == 1)
		debug3("RNG is ready, skipping seeding");
	else {
		if (seed_from_prngd(buf, sizeof(buf)) == -1)
			fatal("Could not obtain seed from PRNGd");
		RAND_add(buf, sizeof(buf), sizeof(buf));
	}
#endif /* OPENSSL_PRNG_ONLY */

	if (RAND_status() != 1)
		fatal("PRNG is not seeded");

	/* Ensure arc4random() is primed */
	arc4random_buf(buf, sizeof(buf));
	explicit_bzero(buf, sizeof(buf));
}

#else /* WITH_OPENSSL */

#include <stdlib.h>
#include <string.h>

/* Actual initialisation is handled in arc4random() */
void
seed_rng(void)
{
	unsigned char buf[RANDOM_SEED_SIZE];

	/* Ensure arc4random() is primed */
	arc4random_buf(buf, sizeof(buf));
	explicit_bzero(buf, sizeof(buf));
}

#endif /* WITH_OPENSSL */
