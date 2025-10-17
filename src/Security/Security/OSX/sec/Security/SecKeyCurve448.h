/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
	@header SecKeyCurve448.h
	The functions provided in SecKeyCurve448.h implement and manage Curve448
    public or private key.
*/

#ifndef _SECURITY_SECKEYCURVE448_H_
#define _SECURITY_SECKEYCURVE448_H_

#include <Security/SecBase.h>
#include <Security/SecKey.h>
#include <Security/SecKeyPriv.h>
#include <CoreFoundation/CFData.h>

__BEGIN_DECLS

/* Given an Ed448 public key in encoded form return a SecKeyRef representing
   that key. Supported encoding are kSecKeyEncodingBytes. */
SecKeyRef SecKeyCreateEd448PublicKey(CFAllocatorRef allocator,
    const uint8_t *keyData, CFIndex keyDataLength,
    SecKeyEncoding encoding);

/* Given an Ed448 private key in encoded form return a SecKeyRef representing
   that key. Supported encoding is 57 bytes of key material. */
SecKeyRef SecKeyCreateEd448PrivateKey(CFAllocatorRef allocator,
    const uint8_t *keyData, CFIndex keyDataLength,
    SecKeyEncoding encoding);

/* Given an X448 public key in encoded form return a SecKeyRef representing
   that key. Supported encoding are kSecKeyEncodingBytes. */
SecKeyRef SecKeyCreateX448PublicKey(CFAllocatorRef allocator,
    const uint8_t *keyData, CFIndex keyDataLength,
    SecKeyEncoding encoding);

/* Given an X448 private key in encoded form return a SecKeyRef representing
   that key. Supported encoding is 56 bytes of key material. */
SecKeyRef SecKeyCreateX448PrivateKey(CFAllocatorRef allocator,
    const uint8_t *keyData, CFIndex keyDataLength,
    SecKeyEncoding encoding);

__END_DECLS

#endif /* !_SECURITY_SECKEYCURVE448_H_ */
