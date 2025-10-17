/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 26, 2021.
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

//
//  SecServerEncryptionSupport.h
//
//

#ifndef _SECURITY_SECSERVERENCRYPTIONSUPPORT_H_
#define _SECURITY_SECSERVERENCRYPTIONSUPPORT_H_

#include <Availability.h>
#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecKey.h>
#include <Security/SecTrust.h>

// Deprecating for security motives (28715251).
// Compatible implementation still available in SecKey with
// kSecKeyAlgorithmECIESEncryptionStandardX963SHA256AESGCM but should also be
// deprecated for the same reason (28496795).

CFDataRef SecCopyEncryptedToServer(SecTrustRef trustedEvaluation, CFDataRef dataToEncrypt, CFErrorRef *error)
    __OSX_AVAILABLE_BUT_DEPRECATED_MSG(__MAC_10_12, __MAC_10_13, __IPHONE_8_0, __IPHONE_11_0, "Migrate to SecKeyCreateEncryptedData with kSecKeyAlgorithmECIESEncryptionStandardVariableIV* or Security Foundation SFIESOperation for improved security (encryption is not compatible)");

//
// For testing
//
/* Caution: These functions take an iOS SecKeyRef. Careful use is required on macOS. */
CFDataRef SecCopyDecryptedForServer(SecKeyRef serverFullKey, CFDataRef encryptedData, CFErrorRef* error)
    API_DEPRECATED("Migrate to SecKeyCreateEncryptedData with kSecKeyAlgorithmECIESEncryptionStandardVariableIV* or Security Foundation SFIESOperation for improved security (encryption is not compatible)", macos(10.12,10.13), ios(8.0,11.0));
// SFIESCiphertext


CFDataRef SecCopyEncryptedToServerKey(SecKeyRef publicKey, CFDataRef dataToEncrypt, CFErrorRef *error)
    __OSX_AVAILABLE_BUT_DEPRECATED_MSG(__MAC_10_12, __MAC_10_13, __IPHONE_8_0, __IPHONE_11_0,"Migrate to SecKeyCreateEncryptedData with kSecKeyAlgorithmECIESEncryptionStandardVariableIV* or Security Foundation SFIESOperation for improved security (encryption is not compatible)");

#endif
