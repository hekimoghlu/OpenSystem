/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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
	@header SecECKey
	The functions provided in SecECKey.h implement and manage an elliptic-curve
    public or private key.
*/

#ifndef _SECURITY_SECECKEY_H_
#define _SECURITY_SECECKEY_H_

#include <Security/SecBase.h>
#include <Security/SecKey.h>
#include <Security/SecKeyPriv.h>
#include <CoreFoundation/CFData.h>

__BEGIN_DECLS

typedef struct SecECPublicKeyParams {
	uint8_t             *modulus;			/* modulus */
	CFIndex             modulusLength;
	uint8_t             *exponent;			/* public exponent */
	CFIndex             exponentLength;
} SecECPublicKeyParams;

/* Given an EC public key in encoded form return a SecKeyRef representing
   that key. Supported encodings are kSecKeyEncodingPkcs1. */
SecKeyRef SecKeyCreateECPublicKey(CFAllocatorRef allocator,
    const uint8_t *keyData, CFIndex keyDataLength,
    SecKeyEncoding encoding);

/* Given an EC private key in encoded form return a SecKeyRef representing
   that key.  Supported encodings are kSecKeyEncodingPkcs1. */
SecKeyRef SecKeyCreateECPrivateKey(CFAllocatorRef allocator,
    const uint8_t *keyData, CFIndex keyDataLength,
    SecKeyEncoding encoding);

__END_DECLS

#endif /* !_SECURITY_SECECKEY_H_ */
