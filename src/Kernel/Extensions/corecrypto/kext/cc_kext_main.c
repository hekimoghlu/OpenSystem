/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 13, 2025.
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
#include <sys/systm.h>
#include <mach/mach_types.h>
#include "register_crypto.h"
#include "prng_random.h"
#include <corecrypto/cckprng.h>
#include "../algorithms/yarrow/yarrow.h"

extern struct crypto_functions pdcrypto_internal_functions;

static const struct cckprng_funcs cc_kprng_fns = {
	.init = cckprng_init,
	.initgen = cckprng_initgen,
	.reseed = cckprng_reseed,
	.refresh = cckprng_refresh,
	.generate = cckprng_generate
};

static struct cckprng_ctx cc_kprng_ctx = {
	.prng = NULL,
	.bytes_since_entropy = 0,
	.bytes_generated = 0
};

kern_return_t cc_kext_start(kmod_info_t * ki, void *d)
{
	int ret = register_crypto_functions(&pdcrypto_internal_functions);
	if (ret == -1) {
		printf("warning: corecrypto could not be registered. Did another crypto handler beat us to it?\n");
	} else {
		prng_error_status error = prngInitialize(&cc_kprng_ctx.prng);
		if (error != PRNG_SUCCESS) {
			panic("prngInitialize() failed with code %d", error);
		}

		register_and_init_prng(&cc_kprng_ctx, &cc_kprng_fns);
		printf("corecrypto loaded\n");
	}

    return KERN_SUCCESS;
}

kern_return_t cc_kext_stop(kmod_info_t *ki, void *d)
{
    return KERN_FAILURE;
}
