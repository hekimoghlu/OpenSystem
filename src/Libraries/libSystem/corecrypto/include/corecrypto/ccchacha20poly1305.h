/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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

#ifndef _CORECRYPTO_CCCHACHA20POLY1305_H_
#define _CORECRYPTO_CCCHACHA20POLY1305_H_

#include <stddef.h>
#include <stdint.h>

#define CCCHACHA20_KEY_NBYTES 0
#define CCCHACHA20_NONCE_NBYTES 0
#define CCPOLY1305_TAG_NBYTES 0

void *ccchacha20poly1305_info();
int ccchacha20(const void *key, const void *nonce, uint32_t counter, size_t dataInLength, const void *dataIn, void *dataOut);
int ccchacha20poly1305_decrypt_oneshot(void *ccchacha20poly1305_info_unknown, const void *key, const void *iv, size_t aDataLen, const void *aData, size_t dataInLength, const void *dataIn, void *dataOut, const void *tagIn);
int ccchacha20poly1305_encrypt_oneshot(void *ccchacha20poly1305_info_unknown, const void *key, const void *iv, size_t aDataLen, const void *aData, size_t dataInLength, const void *dataIn, void *dataOut, void *tagOut);

#endif // _CORECRYPTO_CCCHACHA20POLY1305_H_