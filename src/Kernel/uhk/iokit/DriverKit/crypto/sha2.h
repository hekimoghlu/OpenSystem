/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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
#ifndef _CRYPTO_SHA2_H__
#define _CRYPTO_SHA2_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <corecrypto/ccsha2.h>

/*** SHA-256/384/512 Various Length Definitions ***********************/
#define SHA256_BLOCK_LENGTH             CCSHA256_BLOCK_SIZE
#define SHA256_DIGEST_LENGTH    CCSHA256_OUTPUT_SIZE
#define SHA256_DIGEST_STRING_LENGTH     (SHA256_DIGEST_LENGTH * 2 + 1)
#define SHA384_BLOCK_LENGTH             CCSHA512_BLOCK_SIZE
#define SHA384_DIGEST_LENGTH    CCSHA384_OUTPUT_SIZE
#define SHA384_DIGEST_STRING_LENGTH     (SHA384_DIGEST_LENGTH * 2 + 1)
#define SHA512_BLOCK_LENGTH             CCSHA512_BLOCK_SIZE
#define SHA512_DIGEST_LENGTH    CCSHA512_OUTPUT_SIZE
#define SHA512_DIGEST_STRING_LENGTH     (SHA512_DIGEST_LENGTH * 2 + 1)

typedef struct {
	ccdigest_ctx_decl(CCSHA256_STATE_SIZE, CCSHA256_BLOCK_SIZE, ctx);
} SHA256_CTX;

typedef struct SHA512_CTX {
	ccdigest_ctx_decl(CCSHA512_STATE_SIZE, CCSHA512_BLOCK_SIZE, ctx);
} SHA512_CTX;

typedef SHA512_CTX SHA384_CTX;

/*** SHA-256/384/512 Function Prototypes ******************************/

void SHA256_Init(SHA256_CTX *ctx);
void SHA256_Update(SHA256_CTX *ctx, const void *data, size_t len);
void SHA256_Final(void *digest, SHA256_CTX *ctx);

void SHA384_Init(SHA384_CTX *ctx);
void SHA384_Update(SHA384_CTX *ctx, const void *data, size_t len);
void SHA384_Final(void *digest, SHA384_CTX *ctx);

void SHA512_Init(SHA512_CTX *ctx);
void SHA512_Update(SHA512_CTX *ctx, const void *data, size_t len);
void SHA512_Final(void *digest, SHA512_CTX *ctx);

#ifdef  __cplusplus
}
#endif /* __cplusplus */

#endif /* _CRYPTO_SHA2_H__ */
