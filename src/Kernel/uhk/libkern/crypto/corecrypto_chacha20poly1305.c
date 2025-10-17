/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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
#include <corecrypto/ccchacha20poly1305.h>
#include <libkern/crypto/crypto_internal.h>
#include <libkern/crypto/chacha20poly1305.h>

static ccchacha20poly1305_fns_t
fns(void)
{
	return g_crypto_funcs->ccchacha20poly1305_fns;
}

static const struct ccchacha20poly1305_info *
info(void)
{
	return fns()->info();
}

int
chacha20poly1305_init(chacha20poly1305_ctx *ctx, const uint8_t *key)
{
	return fns()->init(info(), ctx, key);
}

int
chacha20poly1305_reset(chacha20poly1305_ctx *ctx)
{
	return fns()->reset(info(), ctx);
}

int
chacha20poly1305_setnonce(chacha20poly1305_ctx *ctx, const uint8_t *nonce)
{
	return fns()->setnonce(info(), ctx, nonce);
}

int
chacha20poly1305_incnonce(chacha20poly1305_ctx *ctx, uint8_t *nonce)
{
	return fns()->incnonce(info(), ctx, nonce);
}

int
chacha20poly1305_aad(chacha20poly1305_ctx *ctx, size_t nbytes, const void *aad)
{
	return fns()->aad(info(), ctx, nbytes, aad);
}

int
chacha20poly1305_encrypt(chacha20poly1305_ctx *ctx, size_t nbytes, const void *ptext, void *ctext)
{
	return fns()->encrypt(info(), ctx, nbytes, ptext, ctext);
}

int
chacha20poly1305_finalize(chacha20poly1305_ctx *ctx, uint8_t *tag)
{
	return fns()->finalize(info(), ctx, tag);
}

int
chacha20poly1305_decrypt(chacha20poly1305_ctx *ctx, size_t nbytes, const void *ctext, void *ptext)
{
	return fns()->decrypt(info(), ctx, nbytes, ctext, ptext);
}

int
chacha20poly1305_verify(chacha20poly1305_ctx *ctx, const uint8_t *tag)
{
	return fns()->verify(info(), ctx, tag);
}
