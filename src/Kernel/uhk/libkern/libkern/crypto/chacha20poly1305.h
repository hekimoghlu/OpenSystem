/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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
#ifndef _CHACHA20POLY1305_H
#define _CHACHA20POLY1305_H

#if defined(__cplusplus)
extern "C"
{
#endif

#include <corecrypto/ccchacha20poly1305.h>

typedef ccchacha20poly1305_ctx chacha20poly1305_ctx;

int     chacha20poly1305_init(chacha20poly1305_ctx *ctx, const uint8_t *key);
int chacha20poly1305_reset(chacha20poly1305_ctx *ctx);
int chacha20poly1305_setnonce(chacha20poly1305_ctx *ctx, const uint8_t *nonce);
int chacha20poly1305_incnonce(chacha20poly1305_ctx *ctx, uint8_t *nonce);
int     chacha20poly1305_aad(chacha20poly1305_ctx *ctx, size_t nbytes, const void *aad);
int     chacha20poly1305_encrypt(chacha20poly1305_ctx *ctx, size_t nbytes, const void *ptext, void *ctext);
int     chacha20poly1305_finalize(chacha20poly1305_ctx *ctx, uint8_t *tag);
int     chacha20poly1305_decrypt(chacha20poly1305_ctx *ctx, size_t nbytes, const void *ctext, void *ptext);
int     chacha20poly1305_verify(chacha20poly1305_ctx *ctx, const uint8_t *tag);

#if defined(__cplusplus)
}
#endif

#endif
