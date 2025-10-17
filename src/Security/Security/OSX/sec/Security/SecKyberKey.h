/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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
/*!
    @header SecKyberKey.h
    The functions provided in SecKyberKey.h implement and manage Kyber cipher
*/

#ifndef _SECURITY_SECKYBERKEY_H_
#define _SECURITY_SECKYBERKEY_H_

#include <Security/SecBase.h>
#include <Security/SecKey.h>
#include <Security/SecKeyPriv.h>

__BEGIN_DECLS

SecKeyRef SecKeyCreateKyberPublicKey(CFAllocatorRef allocator, const uint8_t *keyData, CFIndex keyDataLength);
SecKeyRef SecKeyCreateKyberPrivateKey(CFAllocatorRef allocator, const uint8_t *keyData, CFIndex keyDataLength);
OSStatus SecKyberKeyGeneratePair(CFDictionaryRef parameters, SecKeyRef *publicKey, SecKeyRef *privateKey);

__END_DECLS

#endif /* !_SECURITY_SECKYBERKEY_H_ */
