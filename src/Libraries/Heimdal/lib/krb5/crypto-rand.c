/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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
#include "krb5_locl.h"

#ifdef __APPLE__
KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_generate_random_block(void *buf, size_t len)
{
    if (CCRandomCopyBytes(kCCRandomDefault, buf, len) != 0)
	krb5_abortx(NULL, "Failed reading %lu random bytes",
		    (unsigned long)len);
}
#else

#define ENTROPY_NEEDED 128

static HEIMDAL_MUTEX crypto_mutex = HEIMDAL_MUTEX_INITIALIZER;

static int
seed_something(void)
{
    char buf[1024], seedfile[256];

    /* If there is a seed file, load it. But such a file cannot be trusted,
       so use 0 for the entropy estimate */

    if (krb5_homedir_access(NULL) && RAND_file_name(seedfile, sizeof(seedfile))) {
	int fd;
	fd = open(seedfile, O_RDONLY | O_BINARY | O_CLOEXEC);
	if (fd >= 0) {
	    ssize_t ret;
	    rk_cloexec(fd);
	    ret = read(fd, buf, sizeof(buf));
	    if (ret > 0)
		RAND_add(buf, ret, 0.0);
	    close(fd);
	} else
	    seedfile[0] = '\0';
    } else
	seedfile[0] = '\0';

    /* Calling RAND_status() will try to use /dev/urandom if it exists so
       we do not have to deal with it. */
    if (RAND_status() != 1) {
#ifndef _WIN32
	krb5_context context;
	const char *p;

	/* Try using egd */
	if (!krb5_init_context(&context)) {
	    p = krb5_config_get_string(context, NULL, "libdefaults",
				       "egd_socket", NULL);
	    if (p != NULL)
		RAND_egd_bytes(p, ENTROPY_NEEDED);
	    krb5_free_context(context);
	}
#else
	/* TODO: Once a Windows CryptoAPI RAND method is defined, we
	   can use that and failover to another method. */
#endif
    }

    if (RAND_status() == 1)	{
	/* Update the seed file */
	if (seedfile[0])
	    RAND_write_file(seedfile);

	return 0;
    } else
	return -1;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_generate_random_block(void *buf, size_t len)
{
    static int rng_initialized = 0;

    HEIMDAL_MUTEX_lock(&crypto_mutex);
    if (!rng_initialized) {
	if (seed_something())
	    krb5_abortx(NULL, "Fatal: could not seed the "
			"random number generator");

	rng_initialized = 1;
    }
    HEIMDAL_MUTEX_unlock(&crypto_mutex);
    if (RAND_bytes(buf, len) <= 0)
	krb5_abortx(NULL, "Failed to generate random block");
}

#endif /* __APPLE__ */
