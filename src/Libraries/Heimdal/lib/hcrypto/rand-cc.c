/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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


#include <stdio.h>
#include <stdlib.h>
#include <rand.h>
#include <heim_threads.h>

#include <roken.h>

#include "randi.h"

#if defined(__APPLE_PRIVATE__) && !defined(__APPLE_TARGET_EMBEDDED__)

#include <CommonCrypto/CommonRandomSPI.h>

/*
 * Unix /dev/random
 */

static void
cc_seed(const void *indata, int size)
{
}


static int
cc_bytes(unsigned char *outdata, int size)
{
    if (CCRandomCopyBytes(kCCRandomDefault, outdata, size) != kCCSuccess)
	return 0;
    return 1;
}

static void
cc_cleanup(void)
{
}

static void
cc_add(const void *indata, int size, double entropi)
{
}

static int
cc_pseudorand(unsigned char *outdata, int size)
{
    return cc_bytes(outdata, size);
}

static int
cc_status(void)
{
    return 1;
}

const RAND_METHOD hc_rand_cc_method = {
    cc_seed,
    cc_bytes,
    cc_cleanup,
    cc_add,
    cc_pseudorand,
    cc_status
};

#endif /* __APPLE_PRIVATE__ */

const RAND_METHOD *
RAND_cc_method(void)
{
#if defined(__APPLE_PRIVATE__) && !defined(__APPLE_TARGET_EMBEDDED__)
    return &hc_rand_cc_method;
#else
    return NULL;
#endif
}
