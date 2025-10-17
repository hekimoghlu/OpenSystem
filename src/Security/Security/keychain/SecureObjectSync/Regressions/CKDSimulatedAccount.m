/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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
//  CKDSimulatedAccount+CKDSimulatedAccount.m
//  Security
//

#import "CKDSimulatedAccount.h"

#import <Foundation/Foundation.h>

@interface CKDSimulatedAccount ()
@property (readwrite) NSMutableDictionary<NSString*, NSObject*>* keyChanges;
@property (readwrite) NSMutableSet<NSString*>* peerChanges;
@property (readwrite) NSMutableSet<NSString*>* backupPeerChanges;
@property (readwrite) BOOL peerRegistrationEnsured;
@end

@implementation CKDSimulatedAccount

+ (instancetype) account {
    return [[CKDSimulatedAccount alloc] init];
}
- (instancetype) init {
    if ((self = [super init])) {
        self.keysToNotHandle = [NSMutableSet<NSString*> set];
        self.keyChanges = [NSMutableDictionary<NSString*, NSObject*> dictionary];

        self.peerChanges = [NSMutableSet<NSString*> set];
        self.peersToNotSyncWith = [NSMutableSet<NSString*> set];

        self.backupPeerChanges = [NSMutableSet<NSString*> set];
        self.backupPeersToNotSyncWith = [NSMutableSet<NSString*> set];

        self.peerRegistrationEnsured = NO;
    }
    return self;
}

- (NSSet*) keysChanged: (NSDictionary<NSString*, NSObject*>*) keyValues
                 error: (NSError**) error {

    NSMutableSet<NSString*>* result = [NSMutableSet<NSString*> set];

    [keyValues enumerateKeysAndObjectsUsingBlock:^(NSString * _Nonnull key, NSObject * _Nonnull obj, BOOL * _Nonnull stop) {
        if (![self.keysToNotHandle containsObject:key]) {
            [self.keyChanges setObject:obj forKey:key];
            [result addObject:key];
        }
    }];

    return result;
}

- (bool) ensurePeerRegistration: (NSError**) error {
    if (self.peerRegistrationFailureReason == nil) {
        self.peerRegistrationEnsured = YES;
        return true;
    } else {
        if (error) {
            *error = self.peerRegistrationFailureReason;
        }
        return false;
    }
}

- (NSSet<NSString*>*) syncWithPeers: (NSSet<NSString*>*) peerIDs
                            backups: (NSSet<NSString*>*) backupPeerIDs
                              error: (NSError**) error {
    NSMutableSet<NSString*>* peerIDsToTake = [peerIDs mutableCopy];
    [peerIDsToTake minusSet:self.peersToNotSyncWith];
    [self.peerChanges unionSet: peerIDsToTake];

    NSMutableSet<NSString*>* backupPeerIDsToTake = [NSMutableSet<NSString*> setWithSet: backupPeerIDs];
    [backupPeerIDsToTake minusSet:self.backupPeersToNotSyncWith];
    [self.backupPeerChanges unionSet: backupPeerIDsToTake];

    // Calculate what we took.
    [peerIDsToTake unionSet:backupPeerIDsToTake];
    return peerIDsToTake;
}

- (SyncWithAllPeersReason) syncWithAllPeers: (NSError**) error {
    return kSyncWithAllPeersSuccess;
}

- (NSDictionary<NSString*, NSObject*>*) extractKeyChanges {
    NSDictionary<NSString*, NSObject*>* result = self.keyChanges;
    self.keyChanges = [NSMutableDictionary<NSString*, NSObject*> dictionary];
    return result;
}

- (NSSet<NSString*>*) extractPeerChanges {
    NSSet<NSString*>* result = self.peerChanges;
    self.peerChanges = [NSMutableSet<NSString*> set];
    return result;
}
- (NSSet<NSString*>*) extractBackupPeerChanges {
    NSSet<NSString*>* result = self.backupPeerChanges;
    self.backupPeerChanges = [NSMutableSet<NSString*> set];
    return result;
}

- (BOOL) extractRegistrationEnsured {
    BOOL result = self.peerRegistrationEnsured;
    self.peerRegistrationEnsured = NO;
    return result;
}

@end
