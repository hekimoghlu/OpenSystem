/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
//  SOSAccountGetSet.c
//  Security
//
//

#include "SOSAccountPriv.h"

#include <utilities/SecCFWrappers.h>

#import "keychain/SecureObjectSync/SOSAccountTrust.h"
#import "keychain/SecureObjectSync/SOSAccountTrustClassic.h"

//
// MARK: Generic Value manipulation
//

static inline bool SOSAccountEnsureExpansion(SOSAccount* account, CFErrorRef *error) {
    
    if (!account.trust.expansion) {
        account.trust.expansion = [NSMutableDictionary dictionary];
    }

    return SecAllocationError(((__bridge CFDictionaryRef)account.trust.expansion), error, CFSTR("Can't Alloc Account Expansion dictionary"));
}

bool SOSAccountClearValue(SOSAccount* account, CFStringRef key, CFErrorRef *error) {
    bool success = SOSAccountEnsureExpansion(account, error);
    if(!success){
        return success;
    }

    [account.trust.expansion removeObjectForKey: (__bridge NSString* _Nonnull)(key)];

    return success;
}

bool SOSAccountSetValue(SOSAccount* account, CFStringRef key, CFTypeRef value, CFErrorRef *error) {
    if (value == NULL) return SOSAccountClearValue(account, key, error);

    bool success = SOSAccountEnsureExpansion(account, error);
    if(!success)
        return success;

    [account.trust.expansion setObject:(__bridge id _Nonnull)(value) forKey:(__bridge NSString* _Nonnull)(key)];

    return success;
}

//
// MARK: UUID
CFStringRef SOSAccountCopyUUID(SOSAccount* account) {
    CFStringRef uuid = CFRetainSafe(asString(SOSAccountGetValue(account, kSOSAccountUUID, NULL), NULL));
    if (uuid == NULL) {
        CFUUIDRef newID = CFUUIDCreate(kCFAllocatorDefault);
        uuid = CFUUIDCreateString(kCFAllocatorDefault, newID);

        CFErrorRef setError = NULL;
        if (!SOSAccountSetValue(account, kSOSAccountUUID, uuid, &setError)) {
            secerror("Failed to set UUID: %@ (%@)", uuid, setError);
        }
        CFReleaseNull(setError);
        CFReleaseNull(newID);
    }
    return uuid;
}

void SOSAccountEnsureUUID(SOSAccount* account) {
    CFStringRef uuid = SOSAccountCopyUUID(account);
    CFReleaseNull(uuid);
}

