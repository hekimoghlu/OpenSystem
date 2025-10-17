/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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
#include "ccGlobals.h"
#include "ccdebug.h"
#include <CommonCrypto/CommonCryptor.h>
#include <CommonCrypto/CommonCryptorSPI.h>
#include "CommonCryptorPriv.h"

CCCryptorStatus CCDesIsWeakKey( void *key, size_t length)
{
    CC_DEBUG_LOG("Entering\n");
    return ccdes_key_is_weak(key, length);
}

void CCDesSetOddParity(void *key, size_t Length)
{
    CC_DEBUG_LOG("Entering\n");
    ccdes_key_set_odd_parity(key, Length);
}

uint32_t CCDesCBCCksum(void *in, void *out, size_t length,
                       void *key, size_t keylen, void *ivec)
{
    CC_DEBUG_LOG("Entering\n");
    return ccdes_cbc_cksum(in, out, length, key, keylen, ivec);
}
