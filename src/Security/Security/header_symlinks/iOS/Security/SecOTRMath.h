/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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
#ifndef _SECOTRMATH_H_
#define _SECOTRMATH_H_

#include <CoreFoundation/CFBase.h>

#include <corecrypto/ccn.h>
#include <corecrypto/ccaes.h>
#include <corecrypto/ccmode.h>

#define kOTRAuthKeyBytes 16
#define kOTRAuthMACKeyBytes 32

#define kOTRMessageKeyBytes 16
#define kOTRMessageMacKeyBytes 20

#define kExponentiationBits 1536
#define kExponentiationUnits ccn_nof(kExponentiationBits)
#define kExponentiationBytes ((kExponentiationBits+7)/8)

#define kSHA256HMAC160Bits  160
#define kSHA256HMAC160Bytes (kSHA256HMAC160Bits/8)

typedef enum {
    kSSID = 0x00,
    kCs = 0x01,
    kM1 = 0x02,
    kM2 = 0x03,
    kM1Prime = 0x04,
    kM2Prime = 0x05
} OTRKeyType;


void DeriveOTR256BitsFromS(OTRKeyType whichKey, size_t sSize, const cc_unit* s, size_t keySize, uint8_t* key);
void DeriveOTR128BitPairFromS(OTRKeyType whichHalf, size_t sSize, const cc_unit* s,
                              size_t firstKeySize, uint8_t* firstKey,
                              size_t secondKeySize, uint8_t* secondKey);
void DeriveOTR64BitsFromS(OTRKeyType whichKey, size_t sSize, const cc_unit* s,
                          size_t firstKeySize, uint8_t* firstKey);


void AES_CTR_HighHalf_Transform(size_t keySize, const uint8_t* key,
                                uint64_t highHalf,
                                size_t howMuch, const uint8_t* from,
                                uint8_t* to);

void AES_CTR_IV0_Transform(size_t keySize, const uint8_t* key,
                           size_t howMuch, const uint8_t* from,
                           uint8_t* to);

#endif
