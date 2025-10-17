/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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
/* $Id$ */

#ifndef HEIM_HMAC_H
#define HEIM_HMAC_H 1

#include <hcrypto/evp.h>

/* symbol renaming */
#define HMAC_CTX_init hc_HMAC_CTX_init
#define HMAC_CTX_cleanup hc_HMAC_CTX_cleanup
#define HMAC_size hc_HMAC_size
#define HMAC_Init_ex hc_HMAC_Init_ex
#define HMAC_Update hc_HMAC_Update
#define HMAC_Final hc_HMAC_Final
#define HMAC hc_HMAC

/*
 *
 */

#define HMAC_MAX_MD_CBLOCK	64

typedef struct hc_HMAC_CTX HMAC_CTX;

struct hc_HMAC_CTX {
    const EVP_MD *md;
    ENGINE *engine;
    EVP_MD_CTX *ctx;
    size_t key_length;
    void *opad;
    void *ipad;
    void *buf;
};


void	HMAC_CTX_init(HMAC_CTX *);
void	HMAC_CTX_cleanup(HMAC_CTX *ctx);

size_t	HMAC_size(const HMAC_CTX *ctx);

void	HMAC_Init_ex(HMAC_CTX *, const void *, size_t,
		     const EVP_MD *, ENGINE *);
void	HMAC_Update(HMAC_CTX *ctx, const void *data, size_t len);
void	HMAC_Final(HMAC_CTX *ctx, void *md, unsigned int *len);

void *	HMAC(const EVP_MD *evp_md, const void *key, size_t key_len,
	     const void *data, size_t n, void *md, unsigned int *md_len);

#endif /* HEIM_HMAC_H */
