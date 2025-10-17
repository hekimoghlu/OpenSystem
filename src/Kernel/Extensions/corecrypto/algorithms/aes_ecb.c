/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 16, 2023.
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
//  Created by rafirafi on 3/17/16.
//  Copyright (c) 2016 rafirafi. All rights reserved.

#if KERNEL
#include <sys/systm.h>
#endif

#include <stddef.h>
#include <corecrypto/ccmode.h>
#include <corecrypto/ccaes.h>
#include <corecrypto/ccn.h>
#include <corecrypto/cc_abort.h>

#include "aes128.h"

static int pdcmode_aes_ecb_init(const struct ccmode_ecb *ecb, ccecb_ctx *ctx, size_t key_len, const void *key)
{
    printf("%s\n", __func__);

    // normalize key lenght
    //  " Key lengths in the range 16 <= key_len <= 32 are given in bytes,
    //   those in the range 128 <= key_len <= 256 are given in bits " xnu/libkern/libkern/crypto/aes.h
    if (key_len > 32) {
        assert(key_len % 8 == 0);
        key_len /= 8;
    }

    // only 128 case implemented here
    if (key_len != CCAES_KEY_SIZE_128) {
        cc_abort("%s key len != 128\n", __func__);
    }

    AES128_set_key((struct _pdcmode_aes128_ctx *)ctx, key);
	return 0;
}

static int pdcmode_aes_ecb_encrypt(const ccecb_ctx *ctx, unsigned long nblocks, const void *in, void *out)
{
    printf("%s\n", __func__);

    AES128_ECB_encrypt((struct _pdcmode_aes128_ctx *)ctx, nblocks, in, out);
	return 0;
}

static int pdcmode_aes_ecb_decrypt(const ccecb_ctx *ctx, unsigned long nblocks, const void *in, void *out)
{
    printf("%s\n", __func__);

    AES128_ECB_decrypt((struct _pdcmode_aes128_ctx *)ctx, nblocks, in, out);
	return 0;
}

const struct ccmode_ecb pdcaes_ecb_encrypt = {
    .size = ccn_sizeof_size(sizeof(struct _pdcmode_aes128_ctx)),
    .block_size = CCAES_BLOCK_SIZE,
    .init = pdcmode_aes_ecb_init,
    .ecb = pdcmode_aes_ecb_encrypt
};

const struct ccmode_ecb pdcaes_ecb_decrypt = {
    .size = ccn_sizeof_size(sizeof(struct _pdcmode_aes128_ctx)),
    .block_size = CCAES_BLOCK_SIZE,
    .init = pdcmode_aes_ecb_init,
    .ecb = pdcmode_aes_ecb_decrypt
};
