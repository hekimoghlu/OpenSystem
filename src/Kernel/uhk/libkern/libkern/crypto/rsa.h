/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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
#ifndef _RSA_H
#define _RSA_H

#if defined(__cplusplus)
extern "C"
{
#endif

#include <corecrypto/ccrsa.h>
#define RSA_MAX_KEY_BITSIZE 4096

typedef struct{
	ccrsa_pub_ctx_decl(ccn_sizeof(RSA_MAX_KEY_BITSIZE), key);
} rsa_pub_ctx;

int rsa_make_pub(rsa_pub_ctx *pub,
    size_t exp_nbytes, const uint8_t *exp,
    size_t mod_nbytes, const uint8_t *mod);

int rsa_verify_pkcs1v15(rsa_pub_ctx *pub, const uint8_t *oid,
    size_t digest_len, const uint8_t *digest,
    size_t sig_len, const uint8_t *sig,
    bool *valid);

#if defined(__cplusplus)
}
#endif

#endif
