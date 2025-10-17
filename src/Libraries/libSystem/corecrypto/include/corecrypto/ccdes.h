/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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
#ifndef _CORECRYPTO_CCDES_H_
#define _CORECRYPTO_CCDES_H_

#include <corecrypto/ccmode.h>

#define CCDES_BLOCK_SIZE 8
#define CCDES_KEY_SIZE 8

extern const struct ccmode_ecb ccdes_ltc_ecb_decrypt_mode;
extern const struct ccmode_ecb ccdes_ltc_ecb_encrypt_mode;

extern const struct ccmode_ecb ccdes3_ltc_ecb_decrypt_mode;
extern const struct ccmode_ecb ccdes3_ltc_ecb_encrypt_mode;
extern const struct ccmode_ecb ccdes168_ltc_ecb_encrypt_mode;

const struct ccmode_ecb *ccdes_ecb_decrypt_mode(void);
const struct ccmode_ecb *ccdes_ecb_encrypt_mode(void);

const struct ccmode_cbc *ccdes_cbc_decrypt_mode(void);
const struct ccmode_cbc *ccdes_cbc_encrypt_mode(void);

const struct ccmode_cfb *ccdes_cfb_decrypt_mode(void);
const struct ccmode_cfb *ccdes_cfb_encrypt_mode(void);

const struct ccmode_cfb8 *ccdes_cfb8_decrypt_mode(void);
const struct ccmode_cfb8 *ccdes_cfb8_encrypt_mode(void);

const struct ccmode_ctr *ccdes_ctr_crypt_mode(void);

const struct ccmode_ofb *ccdes_ofb_crypt_mode(void);


const struct ccmode_ecb *ccdes3_ecb_decrypt_mode(void);
const struct ccmode_ecb *ccdes3_ecb_encrypt_mode(void);

const struct ccmode_cbc *ccdes3_cbc_decrypt_mode(void);
const struct ccmode_cbc *ccdes3_cbc_encrypt_mode(void);

const struct ccmode_cfb *ccdes3_cfb_decrypt_mode(void);
const struct ccmode_cfb *ccdes3_cfb_encrypt_mode(void);

const struct ccmode_cfb8 *ccdes3_cfb8_decrypt_mode(void);
const struct ccmode_cfb8 *ccdes3_cfb8_encrypt_mode(void);

const struct ccmode_ctr *ccdes3_ctr_crypt_mode(void);

const struct ccmode_ofb *ccdes3_ofb_crypt_mode(void);

int ccdes_key_is_weak( void *key, size_t  length);
void ccdes_key_set_odd_parity(void *key, size_t length);

uint32_t
ccdes_cbc_cksum(void *in, void *out, size_t length,
                void *key, size_t keylen, void *ivec);


#endif /* _CORECRYPTO_CCDES_H_ */
