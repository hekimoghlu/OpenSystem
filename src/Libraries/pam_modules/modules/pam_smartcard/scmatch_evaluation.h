/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 13, 2024.
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

#ifndef scmatch_evaluation_h
#define scmatch_evaluation_h
#include <Security/Security.h>
#include <CoreFoundation/CFArray.h>
#include <OpenDirectory/OpenDirectory.h>

SecKeychainRef copyAttributeMatchedKeychain(ODRecordRef odRecord, CFArrayRef identities, SecIdentityRef* returnedIdentity);
SecKeychainRef copyHashMatchedKeychain(ODRecordRef odRecord, CFArrayRef identities, SecIdentityRef* returnedIdentity);

// caller is responsible for releasing copiedIdentity
SecKeychainRef copySmartCardKeychainForUser(ODRecordRef odRecord, const char* username, SecIdentityRef* copiedIdentity);

OSStatus verifySmartCardSigning(SecKeyRef publicKey, SecKeyRef privateKey);
OSStatus validateCertificate(SecCertificateRef certificate, SecKeychainRef keychain);

#define CFReleaseSafe(CF) { CFTypeRef _cf = (CF); if (_cf) CFRelease(_cf); }
#define CFReleaseNull(CF) { CFTypeRef _cf = (CF); \
    if (_cf) { (CF) = NULL; CFRelease(_cf); } }

#endif /* scmatch_evaluation_h */
