/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 25, 2023.
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
#ifndef _SECOTRDHKEY_H_
#define _SECOTRDHKEY_H_

#include <CoreFoundation/CFBase.h>
#include <CoreFoundation/CFData.h>
#include <corecrypto/ccn.h>
#include <corecrypto/ccsha1.h>

__BEGIN_DECLS

typedef struct _SecOTRFullDHKey* SecOTRFullDHKeyRef;
typedef struct _SecOTRPublicDHKey* SecOTRPublicDHKeyRef;

SecOTRFullDHKeyRef SecOTRFullDHKCreate(CFAllocatorRef allocator);
SecOTRFullDHKeyRef SecOTRFullDHKCreateFromBytes(CFAllocatorRef allocator, const uint8_t**bytes, size_t*size);

OSStatus SecFDHKNewKey(SecOTRFullDHKeyRef key);
void SecFDHKAppendSerialization(SecOTRFullDHKeyRef fullKey, CFMutableDataRef appendTo);
void SecFDHKAppendPublicSerialization(SecOTRFullDHKeyRef fullKey, CFMutableDataRef appendTo);
void SecFDHKAppendCompactPublicSerialization(SecOTRFullDHKeyRef fullKey, CFMutableDataRef appendTo);

static const size_t kSecDHKHashSize = CCSHA1_OUTPUT_SIZE;

uint8_t* SecFDHKGetHash(SecOTRFullDHKeyRef pubKey);


SecOTRPublicDHKeyRef SecOTRPublicDHKCreateFromFullKey(CFAllocatorRef allocator, SecOTRFullDHKeyRef full);
SecOTRPublicDHKeyRef SecOTRPublicDHKCreateFromSerialization(CFAllocatorRef allocator, const uint8_t**bytes, size_t*size);
SecOTRPublicDHKeyRef SecOTRPublicDHKCreateFromCompactSerialization(CFAllocatorRef allocator, const uint8_t** bytes, size_t *size);
SecOTRPublicDHKeyRef SecOTRPublicDHKCreateFromBytes(CFAllocatorRef allocator, const uint8_t** bytes, size_t *size);

void SecPDHKAppendSerialization(SecOTRPublicDHKeyRef pubKey, CFMutableDataRef appendTo);
void SecPDHKAppendCompactSerialization(SecOTRPublicDHKeyRef pubKey, CFMutableDataRef appendTo);
uint8_t* SecPDHKGetHash(SecOTRPublicDHKeyRef pubKey);

void SecPDHKeyGenerateS(SecOTRFullDHKeyRef myKey, SecOTRPublicDHKeyRef theirKey, cc_unit* s);

bool SecDHKIsGreater(SecOTRFullDHKeyRef myKey, SecOTRPublicDHKeyRef theirKey);

void SecOTRDHKGenerateOTRKeys(SecOTRFullDHKeyRef myKey, SecOTRPublicDHKeyRef theirKey,
                           uint8_t* sendMessageKey, uint8_t* sendMacKey,
                           uint8_t* receiveMessageKey, uint8_t* receiveMacKey);

__END_DECLS

#endif
