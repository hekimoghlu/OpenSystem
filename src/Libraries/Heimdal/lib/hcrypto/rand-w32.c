/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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
#include <roken.h>

#include <wincrypt.h>

#include <stdio.h>
#include <stdlib.h>
#include <rand.h>
#include <heim_threads.h>

#include "randi.h"

volatile static HCRYPTPROV g_cryptprovider = 0;

static HCRYPTPROV
_hc_CryptProvider(void)
{
    BOOL rv;
    HCRYPTPROV cryptprovider = 0;

    if (g_cryptprovider != 0)
	return g_cryptprovider;

    rv = CryptAcquireContext(&cryptprovider, NULL,
			      MS_ENHANCED_PROV, PROV_RSA_FULL,
			      CRYPT_VERIFYCONTEXT);

    if (GetLastError() == NTE_BAD_KEYSET) {
        rv = CryptAcquireContext(&cryptprovider, NULL,
                                 MS_ENHANCED_PROV, PROV_RSA_FULL,
                                 CRYPT_NEWKEYSET);
    }

    if (rv) {
        /* try the default provider */
        rv = CryptAcquireContext(&cryptprovider, NULL, 0, PROV_RSA_FULL,
                                 CRYPT_VERIFYCONTEXT);

        if (GetLastError() == NTE_BAD_KEYSET) {
            rv = CryptAcquireContext(&cryptprovider, NULL,
                                     MS_ENHANCED_PROV, PROV_RSA_FULL,
                                     CRYPT_NEWKEYSET);
        }
    }

    if (rv) {
        /* try just a default random number generator */
        rv = CryptAcquireContext(&cryptprovider, NULL, 0, PROV_RNG,
                                 CRYPT_VERIFYCONTEXT);
    }

    if (rv &&
        InterlockedCompareExchangePointer((PVOID *) &g_cryptprovider,
					  (PVOID) cryptprovider, 0) != 0) {

        CryptReleaseContext(cryptprovider, 0);
        cryptprovider = g_cryptprovider;
    }

    return cryptprovider;
}

/*
 *
 */


static void
w32crypto_seed(const void *indata, int size)
{
}


static int
w32crypto_bytes(unsigned char *outdata, int size)
{
    if (CryptGenRandom(_hc_CryptProvider(), size, outdata))
	return 1;
    return 0;
}

static void
w32crypto_cleanup(void)
{
    HCRYPTPROV cryptprovider;

    if (InterlockedCompareExchangePointer((PVOID *) &cryptprovider,
					  0, (PVOID) g_cryptprovider) == 0) {
        CryptReleaseContext(cryptprovider, 0);
    }
}

static void
w32crypto_add(const void *indata, int size, double entropi)
{
}

static int
w32crypto_status(void)
{
    if (_hc_CryptProvider() == 0)
	return 0;
    return 1;
}

const RAND_METHOD hc_rand_w32crypto_method = {
    w32crypto_seed,
    w32crypto_bytes,
    w32crypto_cleanup,
    w32crypto_add,
    w32crypto_bytes,
    w32crypto_status
};

const RAND_METHOD *
RAND_w32crypto_method(void)
{
    return &hc_rand_w32crypto_method;
}
