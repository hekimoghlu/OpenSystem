/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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
#ifndef _CRYPTO_DES_H
#define _CRYPTO_DES_H

#ifdef  __cplusplus
extern "C" {
#endif

#include <corecrypto/ccmode.h>
#include <corecrypto/ccdes.h>
#include <corecrypto/ccn.h>

/* must be 32bit quantity */
#define DES_LONG u_int32_t

typedef unsigned char des_cblock[8];

/* Unholy hack: this is currently the size for the only implementation of DES in corecrypto */
#define DES_ECB_CTX_MAX_SIZE (64*4)
#define DES3_ECB_CTX_MAX_SIZE (64*4*3)


typedef struct{
	ccecb_ctx_decl(DES_ECB_CTX_MAX_SIZE, enc);
	ccecb_ctx_decl(DES_ECB_CTX_MAX_SIZE, dec);
} des_ecb_key_schedule;

typedef struct{
	ccecb_ctx_decl(DES3_ECB_CTX_MAX_SIZE, enc);
	ccecb_ctx_decl(DES3_ECB_CTX_MAX_SIZE, dec);
} des3_ecb_key_schedule;

/* Only here for backward compatibility with smb kext */
typedef des_ecb_key_schedule des_key_schedule[1];
#define des_set_key des_ecb_key_sched

#define DES_ENCRYPT     1
#define DES_DECRYPT     0


/* Single DES ECB - 1 block */
int des_ecb_key_sched(des_cblock *key, des_ecb_key_schedule *ks);
int des_ecb_encrypt(des_cblock * in, des_cblock *out, des_ecb_key_schedule *ks, int encrypt);

/* Triple DES ECB - 1 block */
int des3_ecb_key_sched(des_cblock *key, des3_ecb_key_schedule *ks);
int des3_ecb_encrypt(des_cblock *block, des_cblock *, des3_ecb_key_schedule *ks, int encrypt);

int des_is_weak_key(des_cblock *key);

#ifdef  __cplusplus
}
#endif

#endif
