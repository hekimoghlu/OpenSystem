/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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
//  SOSKeyedPubKeyIdentifier.c
//  Security
//

#include "SOSKeyedPubKeyIdentifier.h"
#include "AssertMacros.h"
#include "keychain/SecureObjectSync/SOSInternal.h"
#include <utilities/debugging.h>

#define SEPARATOR CFSTR("-")
#define SEPLOC 2

bool SOSKeyedPubKeyIdentifierIsPrefixed(CFStringRef kpkid) {
    CFRange seploc = CFStringFind(kpkid, SEPARATOR, 0);
    return seploc.location == SEPLOC;
}

static CFStringRef SOSKeyedPubKeyIdentifierCreateWithPrefixAndID(CFStringRef prefix, CFStringRef id) {
    CFMutableStringRef retval = NULL;
    require_quiet(prefix, errOut);
    require_quiet(id, errOut);
    require_quiet(CFStringGetLength(prefix) == SEPLOC, errOut);
    retval = CFStringCreateMutableCopy(kCFAllocatorDefault, 50, prefix);
    CFStringAppend(retval, SEPARATOR);
    CFStringAppend(retval, id);
errOut:
    return retval;
}

CFStringRef SOSKeyedPubKeyIdentifierCreateWithData(CFStringRef prefix, CFDataRef pubKeyData) {
    CFErrorRef localError = NULL;
    CFStringRef id = SOSCopyIDOfDataBuffer(pubKeyData, &localError);
    CFStringRef retval = SOSKeyedPubKeyIdentifierCreateWithPrefixAndID(prefix, id);
    if(!id) secnotice("kpid", "Couldn't create kpid: %@", localError);
    CFReleaseNull(id);
    CFReleaseNull(localError);
    return retval;
}

CFStringRef SOSKeyedPubKeyIdentifierCreateWithSecKey(CFStringRef prefix, SecKeyRef pubKey) {
    CFErrorRef localError = NULL;
    CFStringRef id = SOSCopyIDOfKey(pubKey, &localError);
    CFStringRef retval = SOSKeyedPubKeyIdentifierCreateWithPrefixAndID(prefix, id);
    if(!id) secnotice("kpid", "Couldn't create kpid: %@", localError);
    CFReleaseNull(id);
    CFReleaseNull(localError);
    return retval;
}


CFStringRef SOSKeyedPubKeyIdentifierCopyPrefix(CFStringRef kpkid) {
    CFRange seploc = CFStringFind(kpkid, SEPARATOR, 0);
    if(seploc.location != SEPLOC) return NULL;
    CFRange prefloc = CFRangeMake(0, SEPLOC);
    return CFStringCreateWithSubstring(kCFAllocatorDefault, kpkid, prefloc);
}

CFStringRef SOSKeyedPubKeyIdentifierCopyHpub(CFStringRef kpkid) {
    CFRange seploc = CFStringFind(kpkid, SEPARATOR, 0);
    if(seploc.location != SEPLOC) return NULL;
    CFRange idloc = CFRangeMake(seploc.location+1, CFStringGetLength(kpkid) - (SEPLOC+1));
    return CFStringCreateWithSubstring(kCFAllocatorDefault, kpkid, idloc);
}

