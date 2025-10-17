/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 3, 2025.
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
//  CKDKVSProxy.h
//  ckd-xpc

#import <Foundation/Foundation.h>
#import <dispatch/queue.h>
#import <xpc/xpc.h>
#import <IDS/IDS.h>

#import <utilities/debugging.h>

#import "SOSCloudKeychainConstants.h"
#import "SOSCloudKeychainClient.h"

#import "CKDStore.h"
#import "CKDAccount.h"
#import "CKDLockMonitor.h"
#import "XPCNotificationDispatcher.h"

#define XPROXYSCOPE "proxy"

typedef void (^FreshnessResponseBlock)(bool success, NSError *err);

@interface UbiqitousKVSProxy : NSObject<XPCNotificationListener, CKDLockListener>
{
    id currentiCloudToken;
    int callbackMethod;
}

@property (readonly) NSObject<CKDStore>* store;
@property (readonly) NSObject<CKDAccount>* account;
@property (readonly) NSObject<CKDLockMonitor>* lockMonitor;

@property (readonly) NSURL* persistenceURL;

@property (retain, nonatomic) NSMutableSet *alwaysKeys;
@property (retain, nonatomic) NSMutableSet *firstUnlockKeys;
@property (retain, nonatomic) NSMutableSet *unlockedKeys;

@property (atomic) bool seenKVSStoreChange;


@property (retain, nonatomic) NSMutableSet *pendingKeys;
@property (retain, nonatomic) NSMutableSet *shadowPendingKeys;

@property (retain, nonatomic) NSString *dsid;
@property (retain, nonatomic) NSString *accountUUID;

@property (retain, nonatomic) NSMutableSet<NSString*>* pendingSyncPeerIDs;
@property (retain, nonatomic) NSMutableSet<NSString*>* shadowPendingSyncPeerIDs;

@property (retain, nonatomic) NSMutableSet<NSString*>* pendingSyncBackupPeerIDs;
@property (retain, nonatomic) NSMutableSet<NSString*>* shadowPendingSyncBackupPeerIDs;

@property (atomic) bool ensurePeerRegistration;
@property (atomic) bool ensurePeerRegistrationEnqueuedButNotStarted;

// Another version of ensurePeerRegistration due to legacy code structure
@property (atomic) bool shadowEnsurePeerRegistration;

@property (atomic) bool inCallout;

@property (retain, nonatomic) NSMutableArray<FreshnessResponseBlock> *freshnessCompletions;
@property (atomic) dispatch_time_t nextFreshnessTime;

@property (atomic) dispatch_queue_t calloutQueue;

@property (atomic) dispatch_queue_t ckdkvsproxy_queue;

@property (copy, atomic) dispatch_block_t shadowFlushBlock;


- (NSString *)description;
- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)withAccount:(NSObject<CKDAccount>*) account
                      store:(NSObject<CKDStore>*) store
                lockMonitor:(NSObject<CKDLockMonitor>*) lockMonitor
                persistence:(NSURL*) localPersistence;

- (instancetype)initWithAccount:(NSObject<CKDAccount>*) account
                          store:(NSObject<CKDStore>*) store
                    lockMonitor:(NSObject<CKDLockMonitor>*) lockMonitor
                    persistence:(NSURL*) localPersistence NS_DESIGNATED_INITIALIZER;

// Requests:

- (void)clearStore;
- (void)synchronizeStore;
- (id) objectForKey: (NSString*) key;
- (NSDictionary<NSString *, id>*) copyAsDictionary;
- (void)setObjectsFromDictionary:(NSDictionary<NSString*, NSObject*> *)otherDictionary;
- (void)waitForSynchronization:(void (^)(NSDictionary<NSString*, NSObject*> *results, NSError *err))handler;


// Callbacks from stores when things happen
- (void)storeKeysChanged: (NSSet<NSString*>*) changedKeys initial: (bool) initial;
- (void)storeAccountChanged;

- (void)requestEnsurePeerRegistration;

- (void)requestSyncWithPeerIDs: (NSArray<NSString*>*) peerIDs backupPeerIDs: (NSArray<NSString*>*) backupPeerIDs;
- (BOOL)hasSyncPendingFor: (NSString*) peerID;
- (BOOL)hasPendingKey: (NSString*) keyName;

- (void)registerAtTimeKeys:(NSDictionary*)keyparms;

- (NSSet*) keysForCurrentLockState;
- (void) intersectWithCurrentLockState: (NSMutableSet*) set;

- (NSMutableSet*) pendKeysAndGetNewlyPended: (NSSet*) keysToPend;

- (NSMutableSet*) pendingKeysForCurrentLockState;
- (NSMutableSet*) pendKeysAndGetPendingForCurrentLockState: (NSSet*) startingSet;

- (void)processPendingKeysForCurrentLockState;

- (void)registerKeys: (NSDictionary*)keys forAccount: (NSString*) accountUUID;
- (void)removeKeys: (NSArray*)keys forAccount: (NSString*) accountUUID;

- (void)processKeyChangedEvent:(NSDictionary *)keysChangedInCloud;
- (NSMutableDictionary *)copyValues:(NSSet *)keysOfInterest;

- (void) doAfterFlush: (dispatch_block_t) block;
- (void) calloutWith: (void(^)(NSSet *pending, NSSet* pendingSyncIDs, NSSet* pendingBackupSyncIDs, bool ensurePeerRegistration, dispatch_queue_t queue, void(^done)(NSSet *handledKeys, NSSet *handledSyncs, bool handledEnsurePeerRegistration, NSError* error))) callout;
- (void) sendKeysCallout: (NSSet *(^)(NSSet* pending, NSError **error)) handleKeys;

- (void)perfCounters:(void(^)(NSDictionary *counters))callback;

@end
