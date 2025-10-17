/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 11, 2023.
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
#include "config.h"


#define HC_DEPRECATED

#ifdef KRB5
#include <krb5-types.h>
#endif
#include <stdlib.h>

#include <des.h>
#include <rand.h>

#undef __attribute__
#define __attribute__(X)

void HC_DEPRECATED
DES_rand_data(void *outdata, int size)
{
    RAND_bytes(outdata, size);
}

void HC_DEPRECATED
DES_generate_random_block(DES_cblock *block)
{
    RAND_bytes(block, sizeof(*block));
}

#define DES_rand_data_key hc_DES_rand_data_key

void HC_DEPRECATED
DES_rand_data_key(DES_cblock *key);

/*
 * Generate a random DES key.
 */

void HC_DEPRECATED
DES_rand_data_key(DES_cblock *key)
{
    DES_new_random_key(key);
}

void HC_DEPRECATED
DES_set_sequence_number(void *ll)
{
}

void HC_DEPRECATED
DES_set_random_generator_seed(DES_cblock *seed)
{
    RAND_seed(seed, sizeof(*seed));
}

/**
 * Generate a random des key using a random block, fixup parity and
 * skip weak keys.
 *
 * @param key is set to a random key.
 *
 * @return 0 on success, non zero on random number generator failure.
 *
 * @ingroup hcrypto_des
 */

int HC_DEPRECATED
DES_new_random_key(DES_cblock *key)
{
    do {
	if (RAND_bytes(key, sizeof(*key)) != 1)
	    return 1;
	DES_set_odd_parity(key);
    } while(DES_is_weak_key(key));

    return(0);
}

/**
 * Seed the random number generator. Deprecated, use @ref page_rand
 *
 * @param seed a seed to seed that random number generate with.
 *
 * @ingroup hcrypto_des
 */

void HC_DEPRECATED
DES_init_random_number_generator(DES_cblock *seed)
{
    RAND_seed(seed, sizeof(*seed));
}

/**
 * Generate a random key, deprecated since it doesn't return an error
 * code, use DES_new_random_key().
 *
 * @param key is set to a random key.
 *
 * @ingroup hcrypto_des
 */

void HC_DEPRECATED
DES_random_key(DES_cblock *key)
{
    if (DES_new_random_key(key))
	abort();
}
