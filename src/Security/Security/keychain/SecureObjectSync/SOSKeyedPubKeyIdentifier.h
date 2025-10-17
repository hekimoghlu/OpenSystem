/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
//  SOSKeyedPubKeyIdentifier.h
//  Security
//

#ifndef SOSKeyedPubKeyIdentifier_h
#define SOSKeyedPubKeyIdentifier_h

#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecKey.h>

// Simple Prefix on Hash of PubKey strings - 2 characters and a dash
// RK-xxxxxxxxxxxxxxxxxxx...
bool SOSKeyedPubKeyIdentifierIsPrefixed(CFStringRef kpkid);
CFStringRef SOSKeyedPubKeyIdentifierCreateWithData(CFStringRef prefix, CFDataRef pubKeyData);
CFStringRef SOSKeyedPubKeyIdentifierCreateWithSecKey(CFStringRef prefix, SecKeyRef pubKey);
CFStringRef SOSKeyedPubKeyIdentifierCopyPrefix(CFStringRef kpkid);
CFStringRef SOSKeyedPubKeyIdentifierCopyHpub(CFStringRef kpid);



#endif /* SOSKeyedPubKeyIdentifier_h */
