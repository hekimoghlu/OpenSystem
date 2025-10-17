/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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
#include <sys/types.h>
#include <stdio.h>
#include <unistd.h>
#include <CommonCrypto/CommonDigest.h>
#include <CommonCrypto/CommonHMAC.h>
#include <roken.h>
#include <hex.h>
#include "heim-auth.h"
#include "ntlm_err.h"

char *
heim_generate_challenge(const char *hostname)
{
    char host[MAXHOSTNAMELEN], *str = NULL;
    uint32_t num, t;

    if (hostname == NULL) {
	if (gethostname(host, sizeof(host)))
	    return NULL;
	hostname = host;
    }

    t = (uint32_t)time(NULL);
    num = rk_random();
    
    asprintf(&str, "<%lu%lu@%s>", (unsigned long)t,
	     (unsigned long)num, hostname);

    return str;
}

char *
heim_apop_create(const char *challenge, const char *password)
{
    char *str = NULL;
    uint8_t hash[CC_MD5_DIGEST_LENGTH];
    CC_MD5_CTX ctx;

    CC_MD5_Init(&ctx);
    CC_MD5_Update(&ctx, challenge, (CC_LONG)strlen(challenge));
    CC_MD5_Update(&ctx, password, (CC_LONG)strlen(password));

    CC_MD5_Final(hash, &ctx);

    hex_encode_lower(hash, sizeof(hash), &str);

    return str;
}

int
heim_apop_verify(const char *challenge, const char *password, const char *response)
{
    char *str;
    int res;

    str = heim_apop_create(challenge, password);
    if (str == NULL)
	return ENOMEM;

    res = (strcasecmp(str, response) != 0);
    free(str);

    if (res)
	return HNTLM_ERR_INVALID_APOP;
    return 0;
}

struct heim_cram_md5_data {
    CC_MD5_CTX ipad;
    CC_MD5_CTX opad;
};


void
heim_cram_md5_export(const char *password, heim_CRAM_MD5_STATE *state)
{
    size_t keylen = strlen(password);
    uint8_t key[CC_MD5_BLOCK_BYTES];
    uint8_t pad[CC_MD5_BLOCK_BYTES];
    struct heim_cram_md5_data ctx;
    size_t n;

    memset(&ctx, 0, sizeof(ctx));

    if (keylen > CC_MD5_BLOCK_BYTES) {
	CC_MD5(password, (CC_LONG)keylen, key);
	keylen = sizeof(keylen);
    } else {
	memcpy(key, password, keylen);
    }

    memset(pad, 0x36, sizeof(pad));
    for (n = 0; n < keylen; n++)
	pad[n] ^= key[n];

    CC_MD5_Init(&ctx.ipad);
    CC_MD5_Init(&ctx.opad);

    CC_MD5_Update(&ctx.ipad, pad, sizeof(pad));

    memset(pad, 0x5c, sizeof(pad));
    for (n = 0; n < keylen; n++)
	pad[n] ^= key[n];

    CC_MD5_Update(&ctx.opad, pad, sizeof(pad));

    memset(pad, 0, sizeof(pad));
    memset(key, 0, sizeof(key));

    state->istate[0] = htonl(ctx.ipad.A);
    state->istate[1] = htonl(ctx.ipad.B);
    state->istate[2] = htonl(ctx.ipad.C);
    state->istate[3] = htonl(ctx.ipad.D);

    state->ostate[0] = htonl(ctx.opad.A);
    state->ostate[1] = htonl(ctx.opad.B);
    state->ostate[2] = htonl(ctx.opad.C);
    state->ostate[3] = htonl(ctx.opad.D);

    memset(&ctx, 0, sizeof(ctx));
}


heim_cram_md5
heim_cram_md5_import(void *data, size_t len)
{
    heim_CRAM_MD5_STATE state;
    heim_cram_md5 ctx;
    
    if (len != sizeof(state))
	return NULL;

    ctx = calloc(1, sizeof(*ctx));
    if (ctx == NULL)
	return NULL;

    memcpy(&state, data, sizeof(state));

    ctx->ipad.A = ntohl(state.istate[0]);
    ctx->ipad.B = ntohl(state.istate[1]);
    ctx->ipad.C = ntohl(state.istate[2]);
    ctx->ipad.D = ntohl(state.istate[3]);

    ctx->opad.A = ntohl(state.ostate[0]);
    ctx->opad.B = ntohl(state.ostate[1]);
    ctx->opad.C = ntohl(state.ostate[2]);
    ctx->opad.D = ntohl(state.ostate[3]);

    ctx->ipad.Nl = ctx->opad.Nl = 512;
    ctx->ipad.Nh = ctx->opad.Nh = 0;
    ctx->ipad.num = ctx->opad.num = 0;

    return ctx;
}

int
heim_cram_md5_verify_ctx(heim_cram_md5 ctx, const char *challenge, const char *response)
{
    uint8_t hash[CC_MD5_DIGEST_LENGTH];
    char *str = NULL;
    char *response_lower = NULL;
    size_t len;
    int res;

    CC_MD5_Update(&ctx->ipad, challenge, (CC_LONG)strlen(challenge));
    CC_MD5_Final(hash, &ctx->ipad);

    CC_MD5_Update(&ctx->opad, hash, sizeof(hash));
    CC_MD5_Final(hash, &ctx->opad);

    hex_encode_lower(hash, sizeof(hash), &str);
    if (str == NULL)
	return ENOMEM;

    len = strlen(response);
    if (len != sizeof(hash) * 2) {
	return HNTLM_ERR_INVALID_CRAM_MD5;
    }

    response_lower = strdup(response);
    if (response_lower == NULL) {
	free(str);
	return ENOMEM;
    }
    strlwr(response_lower);

    res = ct_memcmp(str, response_lower, len);
    free(response_lower);
    free(str);

    if (res)
	return HNTLM_ERR_INVALID_CRAM_MD5;
    return 0;
}

void
heim_cram_md5_free(heim_cram_md5 ctx)
{
    memset(ctx, 0, sizeof(*ctx));
    free(ctx);
}


char *
heim_cram_md5_create(const char *challenge, const char *password)
{
    CCHmacContext ctx;
    uint8_t hash[CC_MD5_DIGEST_LENGTH];
    char *str = NULL;

    CCHmacInit(&ctx, kCCHmacAlgMD5, password, strlen(password));
    CCHmacUpdate(&ctx, challenge, strlen(challenge));
    CCHmacFinal(&ctx, hash);

    memset(&ctx, 0, sizeof(ctx));

    hex_encode_lower(hash, sizeof(hash), &str);

    return str;
}

 int
heim_cram_md5_verify(const char *challenge, const char *password, const char *response)
{
    char  *rresponse = NULL;
    size_t len;
    char *str;
    int res;

    str = heim_cram_md5_create(challenge, password);
    if (str == NULL)
	return ENOMEM;

    len = strlen(str);
    if (len != strlen(response)) {
	free(str);
	return HNTLM_ERR_INVALID_CRAM_MD5;
    }

    rresponse = strdup(response);
    if (rresponse == NULL) {
	free(str);
	return ENOMEM;
    }
    strlwr(rresponse);

    res = ct_memcmp(str, response, len);
    free(rresponse);
    free(str);

    if (res != 0)
	return HNTLM_ERR_INVALID_CRAM_MD5;
    return 0;
}

