/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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

#ifndef _CORECRYPTO_CCBLOWFISH_H_
#define _CORECRYPTO_CCBLOWFISH_H_

#include <corecrypto/ccmode.h>

#define CCBLOWFISH_BLOCK_SIZE 8

#define CCBLOWFISH_KEY_SIZE_MIN 8
#define CCBLOWFISH_KEY_SIZE_MAX 56

extern const struct ccmode_ecb ccblowfish_ltc_ecb_decrypt_mode;
extern const struct ccmode_ecb ccblowfish_ltc_ecb_encrypt_mode;

const struct ccmode_ecb *ccblowfish_ecb_decrypt_mode();
const struct ccmode_ecb *ccblowfish_ecb_encrypt_mode();

const struct ccmode_cbc *ccblowfish_cbc_decrypt_mode();
const struct ccmode_cbc *ccblowfish_cbc_encrypt_mode();

const struct ccmode_cfb *ccblowfish_cfb_decrypt_mode();
const struct ccmode_cfb *ccblowfish_cfb_encrypt_mode();

const struct ccmode_cfb8 *ccblowfish_cfb8_decrypt_mode();
const struct ccmode_cfb8 *ccblowfish_cfb8_encrypt_mode();

const struct ccmode_ctr *ccblowfish_ctr_crypt_mode();

const struct ccmode_ofb *ccblowfish_ofb_crypt_mode();

#endif

