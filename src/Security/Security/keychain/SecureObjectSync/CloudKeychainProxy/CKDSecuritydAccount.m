/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
//  CKDSecuritydAccount+CKDSecuritydAccount_m.m
//  Security
//
//

#import "Foundation/Foundation.h"
#import "CKDSecuritydAccount.h"

#include <Security/SecureObjectSync/SOSCloudCircleInternal.h>
#include <Security/SecItemPriv.h>

@implementation CKDSecuritydAccount

+ (instancetype) securitydAccount
{
    return [[CKDSecuritydAccount alloc] init];
}

- (NSSet*) keysChanged: (NSDictionary<NSString*, NSObject*>*)keyValues error: (NSError**) error
{
    CFErrorRef cf_error = NULL;
    NSArray* handled = (__bridge_transfer NSArray*) _SecKeychainSyncUpdateMessage((__bridge CFDictionaryRef)keyValues, &cf_error);
    NSError *updateError = (__bridge_transfer NSError*)cf_error;
    if (error)
        *error = updateError;

    return [NSSet setWithArray:handled];
}

- (bool) ensurePeerRegistration: (NSError**) error
{
    CFErrorRef localError = NULL;
    bool result = SOSCCProcessEnsurePeerRegistration(error ? &localError : NULL);

    if (error && localError) {
        *error = (__bridge_transfer NSError*) localError;
    }

    return result;
}

- (NSSet<NSString*>*) syncWithPeers: (NSSet<NSString*>*) peerIDs backups: (NSSet<NSString*>*) backupPeerIDs error: (NSError**) error
{
    CFErrorRef localError = NULL;
    CFSetRef handledPeers = SOSCCProcessSyncWithPeers((__bridge CFSetRef) peerIDs, (__bridge CFSetRef) backupPeerIDs, &localError);

    if (error && localError) {
        *error = (__bridge_transfer NSError*) localError;
    }

    return (__bridge_transfer NSSet<NSString*>*) handledPeers;
}

- (SyncWithAllPeersReason) syncWithAllPeers: (NSError**) error
{
    CFErrorRef localError = NULL;
    SyncWithAllPeersReason result = SOSCCProcessSyncWithAllPeers(error ? &localError : NULL);

    if (error && localError) {
        *error = (__bridge_transfer NSError*) localError;
    }

    return result;
}

@end
