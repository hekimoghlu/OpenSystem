/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 9, 2022.
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

#ifndef _CORECRYPTO_CCCAST_H_
#define _CORECRYPTO_CCCAST_H_

#include <corecrypto/ccmode.h>

#define CCCAST_BLOCK_SIZE 	8
#define CCCAST_KEY_LENGTH 	16
#define CCCAST_MIN_KEY_LENGTH 	5

const struct ccmode_ecb *cccast_ecb_decrypt_mode();
const struct ccmode_ecb *cccast_ecb_encrypt_mode();

const struct ccmode_cbc *cccast_cbc_decrypt_mode();
const struct ccmode_cbc *cccast_cbc_encrypt_mode();

const struct ccmode_cfb *cccast_cfb_decrypt_mode();
const struct ccmode_cfb *cccast_cfb_encrypt_mode();

const struct ccmode_cfb8 *cccast_cfb8_decrypt_mode();
const struct ccmode_cfb8 *cccast_cfb8_encrypt_mode();

const struct ccmode_ctr *cccast_ctr_crypt_mode();

const struct ccmode_ofb *cccast_ofb_crypt_mode();

extern const struct ccmode_ecb cccast_eay_ecb_decrypt_mode;
extern const struct ccmode_ecb cccast_eay_ecb_encrypt_mode;

#endif
