/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 21, 2024.
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

/* compatibility with old or broken OpenSSL versions */
#include "openbsd-compat/openssl-compat.h"

#ifdef USE_BUILTIN_RIJNDAEL
#include <sys/types.h>

#include <openssl/evp.h>

#include <stdarg.h>
#include <string.h>

#include "rijndael.h"
#include "xmalloc.h"
#include "log.h"

#define RIJNDAEL_BLOCKSIZE 16
struct ssh_rijndael_ctx
{
	rijndael_ctx	r_ctx;
	u_char		r_iv[RIJNDAEL_BLOCKSIZE];
};

static int
ssh_rijndael_init(EVP_CIPHER_CTX *ctx, const u_char *key, const u_char *iv,
    int enc)
{
	struct ssh_rijndael_ctx *c;

	if ((c = EVP_CIPHER_CTX_get_app_data(ctx)) == NULL) {
		c = xmalloc(sizeof(*c));
		EVP_CIPHER_CTX_set_app_data(ctx, c);
	}
	if (key != NULL) {
		if (enc == -1)
			enc = ctx->encrypt;
		rijndael_set_key(&c->r_ctx, (u_char *)key,
		    8*EVP_CIPHER_CTX_key_length(ctx), enc);
	}
	if (iv != NULL)
		memcpy(c->r_iv, iv, RIJNDAEL_BLOCKSIZE);
	return (1);
}

static int
ssh_rijndael_cbc(EVP_CIPHER_CTX *ctx, u_char *dest, const u_char *src,
    size_t len)
{
	struct ssh_rijndael_ctx *c;
	u_char buf[RIJNDAEL_BLOCKSIZE];
	u_char *cprev, *cnow, *plain, *ivp;
	int i, j, blocks = len / RIJNDAEL_BLOCKSIZE;

	if (len == 0)
		return (1);
	if (len % RIJNDAEL_BLOCKSIZE)
		fatal("ssh_rijndael_cbc: bad len %d", len);
	if ((c = EVP_CIPHER_CTX_get_app_data(ctx)) == NULL) {
		error("ssh_rijndael_cbc: no context");
		return (0);
	}
	if (ctx->encrypt) {
		cnow  = dest;
		plain = (u_char *)src;
		cprev = c->r_iv;
		for (i = 0; i < blocks; i++, plain+=RIJNDAEL_BLOCKSIZE,
		    cnow+=RIJNDAEL_BLOCKSIZE) {
			for (j = 0; j < RIJNDAEL_BLOCKSIZE; j++)
				buf[j] = plain[j] ^ cprev[j];
			rijndael_encrypt(&c->r_ctx, buf, cnow);
			cprev = cnow;
		}
		memcpy(c->r_iv, cprev, RIJNDAEL_BLOCKSIZE);
	} else {
		cnow  = (u_char *) (src+len-RIJNDAEL_BLOCKSIZE);
		plain = dest+len-RIJNDAEL_BLOCKSIZE;

		memcpy(buf, cnow, RIJNDAEL_BLOCKSIZE);
		for (i = blocks; i > 0; i--, cnow-=RIJNDAEL_BLOCKSIZE,
		    plain-=RIJNDAEL_BLOCKSIZE) {
			rijndael_decrypt(&c->r_ctx, cnow, plain);
			ivp = (i == 1) ? c->r_iv : cnow-RIJNDAEL_BLOCKSIZE;
			for (j = 0; j < RIJNDAEL_BLOCKSIZE; j++)
				plain[j] ^= ivp[j];
		}
		memcpy(c->r_iv, buf, RIJNDAEL_BLOCKSIZE);
	}
	return (1);
}

static int
ssh_rijndael_cleanup(EVP_CIPHER_CTX *ctx)
{
	struct ssh_rijndael_ctx *c;

	if ((c = EVP_CIPHER_CTX_get_app_data(ctx)) != NULL) {
		memset(c, 0, sizeof(*c));
		free(c);
		EVP_CIPHER_CTX_set_app_data(ctx, NULL);
	}
	return (1);
}

void
ssh_rijndael_iv(EVP_CIPHER_CTX *evp, int doset, u_char * iv, u_int len)
{
	struct ssh_rijndael_ctx *c;

	if ((c = EVP_CIPHER_CTX_get_app_data(evp)) == NULL)
		fatal("ssh_rijndael_iv: no context");
	if (doset)
		memcpy(c->r_iv, iv, len);
	else
		memcpy(iv, c->r_iv, len);
}

const EVP_CIPHER *
evp_rijndael(void)
{
	static EVP_CIPHER rijndal_cbc;

	memset(&rijndal_cbc, 0, sizeof(EVP_CIPHER));
	rijndal_cbc.nid = NID_undef;
	rijndal_cbc.block_size = RIJNDAEL_BLOCKSIZE;
	rijndal_cbc.iv_len = RIJNDAEL_BLOCKSIZE;
	rijndal_cbc.key_len = 16;
	rijndal_cbc.init = ssh_rijndael_init;
	rijndal_cbc.cleanup = ssh_rijndael_cleanup;
	rijndal_cbc.do_cipher = ssh_rijndael_cbc;
#ifndef SSH_OLD_EVP
	rijndal_cbc.flags = EVP_CIPH_CBC_MODE | EVP_CIPH_VARIABLE_LENGTH |
	    EVP_CIPH_ALWAYS_CALL_INIT | EVP_CIPH_CUSTOM_IV;
#endif
	return (&rijndal_cbc);
}
#endif /* USE_BUILTIN_RIJNDAEL */
