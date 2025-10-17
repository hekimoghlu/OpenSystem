/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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
	@header SecRSAKey
	The functions provided in SecRSAKey.h implement and manage a rsa
    public or private key.
*/

#ifndef _SECURITY_SECRSAKEY_H_
#define _SECURITY_SECRSAKEY_H_

#include <Security/SecKey.h>
#include <Security/SecKeyPriv.h>
#include <CoreFoundation/CFData.h>

__BEGIN_DECLS

/* Given an RSA public key in encoded form return a SecKeyRef representing
   that key. Supported encodings are kSecKeyEncodingPkcs1. */
SecKeyRef SecKeyCreateRSAPublicKey(CFAllocatorRef allocator,
    const uint8_t *keyData, CFIndex keyDataLength,
    SecKeyEncoding encoding);

CFDataRef SecKeyCopyModulus(SecKeyRef rsaPublicKey);
CFDataRef SecKeyCopyExponent(SecKeyRef rsaPublicKey);

/* Given an RSA private key in encoded form return a SecKeyRef representing
   that key.  Supported encodings are kSecKeyEncodingPkcs1. */
SecKeyRef SecKeyCreateRSAPrivateKey(CFAllocatorRef allocator,
    const uint8_t *keyData, CFIndex keyDataLength,
    SecKeyEncoding encoding);

__END_DECLS

#endif /* !_SECURITY_SECRSAKEY_H_ */
