/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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

#ifndef _AES_H_
#define _AES_H_

#include <stdint.h>

struct _pdcmode_aes128_ctx {
    uint8_t RoundKey[176];
};

struct pdccbc_iv {
    uint8_t b[16];
};

void AES128_set_key(struct _pdcmode_aes128_ctx *ctx, const void *key);

void AES128_ECB_encrypt(const struct _pdcmode_aes128_ctx *ctx, unsigned long nblocks, const void *in, void *out);
void AES128_ECB_decrypt(const struct _pdcmode_aes128_ctx *ctx, unsigned long nblocks, const void *in, void *out);

void AES128_CBC_encrypt(const struct _pdcmode_aes128_ctx *ctx, struct pdccbc_iv* iv, unsigned long nblocks, const void *in, void *out);
void AES128_CBC_decrypt(const struct _pdcmode_aes128_ctx *ctx, struct pdccbc_iv* iv, unsigned long nblocks, const void *in, void *out);

#endif //_AES_H_
