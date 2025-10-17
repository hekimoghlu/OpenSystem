/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
#import "keychain/ckks/CKKSKeychainView.h"
#import "keychain/ckks/CKKSGroupOperation.h"
#import "keychain/ckks/CKKSLocalSynchronizeOperation.h"
#import "keychain/ckks/CKKSFetchAllRecordZoneChangesOperation.h"
#import "keychain/ckks/CKKSScanLocalItemsOperation.h"
#import "keychain/ckks/CKKSMirrorEntry.h"
#import "keychain/ckks/CKKSIncomingQueueEntry.h"
#import "keychain/ckks/CloudKitCategories.h"
#import "keychain/categories/NSError+UsefulConstructors.h"
#import "keychain/ot/ObjCImprovements.h"

#if OCTAGON

@interface CKKSLocalSynchronizeOperation ()
@property int32_t restartCount;
@end

@implementation CKKSLocalSynchronizeOperation

- (instancetype)init {
    return nil;
}
- (instancetype)initWithCKKSKeychainView:(CKKSKeychainView*)ckks
                   operationDependencies:(CKKSOperationDependencies*)operationDependencies
{
    if(self = [super init]) {
        _ckks = ckks;
        _restartCount = 0;
        _deps = operationDependencies;

        [self addNullableDependency:ckks.holdLocalSynchronizeOperation];
    }
    return self;
}

- (void)groupStart {
    WEAKIFY(self);
#if TARGET_OS_TV
    [self.deps.personaAdapter prepareThreadForKeychainAPIUseForPersonaIdentifier: nil];
#endif
    /*
     * A local synchronize is very similar to a CloudKit synchronize, but it won't cause any (non-essential)
     * CloudKit operations to occur.
     *
     *  1. Finish processing the outgoing queue. You can't be in-sync with cloudkit if you have an update that hasn't propagated.
     *  2. Process anything in the incoming queue as normal.
     *          (Note that this might require the keybag to be unlocked.)
     *
     *  3. Take every item in the CKMirror, and check for its existence in the local keychain. If not present, add to the incoming queue.
     *  4. Process the incoming queue again.
     *  5. Scan the local keychain for items which exist locally but are not in CloudKit. Upload them.
     *  6. If there are any such items in 4, restart the sync.
     */

    CKKSKeychainView* ckks = self.ckks;

    // Synchronous, on some thread. Get back on the CKKS queue for SQL thread-safety.
    [ckks dispatchSyncWithSQLTransaction:^CKKSDatabaseTransactionResult{
        if(self.cancelled) {
            ckksnotice("ckksresync", ckks, "CKKSSynchronizeOperation cancelled, quitting");
            return CKKSDatabaseTransactionRollback;
        }

        //ckks.lastLocalSynchronizeOperation = self;

        uint32_t steps = 5;

        ckksinfo("ckksresync", ckks, "Beginning local resynchronize (attempt %u)", self.restartCount);

        CKOperationGroup* operationGroup = [CKOperationGroup CKKSGroupWithName:@"ckks-resync-local"];

        // Step 1
        CKKSOutgoingQueueOperation* outgoingOp = [[CKKSOutgoingQueueOperation alloc] initWithDependencies:ckks.operationDependencies
                                                                                                intending:CKKSStateReady
                                                                                             ckErrorState:CKKSStateOutgoingQueueOperationFailed
                                                                                               errorState:CKKSStateError];
        outgoingOp.name = [NSString stringWithFormat: @"resync-step%u-outgoing", self.restartCount * steps + 1];
        [self runBeforeGroupFinished:outgoingOp];

        // Step 2
        CKKSIncomingQueueOperation* incomingOp = [[CKKSIncomingQueueOperation alloc] initWithDependencies:ckks.operationDependencies
                                                                                                intending:CKKSStateReady
                                                                         pendingClassAItemsRemainingState:SecCKKSZoneKeyStateReady
                                                                                               errorState:CKKSStateUnhealthy
                                                                                handleMismatchedViewItems:false];

        incomingOp.name = [NSString stringWithFormat: @"resync-step%u-incoming", self.restartCount * steps + 2];
        [incomingOp addSuccessDependency:outgoingOp];
        [self runBeforeGroupFinished:incomingOp];

        // Step 3:
        CKKSResultOperation* reloadOp = [[CKKSReloadAllItemsOperation alloc] initWithOperationDependencies:ckks.operationDependencies];
        reloadOp.name = [NSString stringWithFormat: @"resync-step%u-reload", self.restartCount * steps + 3];
        [self runBeforeGroupFinished:reloadOp];

        // Step 4
        CKKSIncomingQueueOperation* incomingResyncOp = [[CKKSIncomingQueueOperation alloc] initWithDependencies:ckks.operationDependencies
                                                                                                      intending:CKKSStateReady
                                                                               pendingClassAItemsRemainingState:SecCKKSZoneKeyStateReady
                                                                                                     errorState:CKKSStateUnhealthy
                                                                                      handleMismatchedViewItems:false];

        incomingResyncOp.name = [NSString stringWithFormat: @"resync-step%u-incoming-again", self.restartCount * steps + 4];
        [incomingResyncOp addSuccessDependency: reloadOp];
        [self runBeforeGroupFinished:incomingResyncOp];

        // Step 5
        CKKSScanLocalItemsOperation* scan = [[CKKSScanLocalItemsOperation alloc] initWithDependencies:ckks.operationDependencies
                                                                                            intending:CKKSStateReady
                                                                                           errorState:CKKSStateError
                                                                                     ckoperationGroup:operationGroup];
        scan.name = [NSString stringWithFormat: @"resync-step%u-scan", self.restartCount * steps + 5];
        [scan addSuccessDependency: incomingResyncOp];
        [self runBeforeGroupFinished: scan];

        // Step 6
        CKKSResultOperation* restart = [[CKKSResultOperation alloc] init];
        restart.name = [NSString stringWithFormat: @"resync-step%u-consider-restart", self.restartCount * steps + 6];
        [restart addExecutionBlock:^{
            STRONGIFY(self);
            if(!self) {
                ckkserror("ckksresync", ckks, "received callback for released object");
                return;
            }

            NSError* error = nil;

            NSMutableSet<CKRecordZoneID*>* ids = [NSMutableSet set];
            for(CKKSKeychainViewState* viewState in ckks.operationDependencies.activeManagedViews) {
                [ids addObject:viewState.zoneID];
            }

            NSSet<NSString*>* iqes = [CKKSIncomingQueueEntry allUUIDsWithContextID:ckks.operationDependencies.contextID inZones:ids error:&error];
            if(error) {
                ckkserror("ckksresync", ckks, "Couldn't fetch IQEs: %@", error);
            }

            if(scan.recordsFound > 0 || iqes.count > 0) {
                if(self.restartCount >= 3) {
                    // we've restarted too many times. Fail and stop.
                    ckkserror("ckksresync", ckks, "restarted synchronization too often; Failing");
                    self.error = [NSError errorWithDomain:@"securityd"
                                                           code:2
                                                       userInfo:@{NSLocalizedDescriptionKey: @"resynchronization restarted too many times; churn in database?"}];
                } else {
                    // restart the sync operation.
                    self.restartCount += 1;
                    ckkserror("ckksresync", ckks, "restarting synchronization operation due to new local items");
                    [self groupStart];
                }
            }
        }];

        [restart addSuccessDependency: scan];
        [self runBeforeGroupFinished: restart];

        return CKKSDatabaseTransactionCommit;
    }];
}

@end;

#pragma mark - CKKSReloadAllItemsOperation

@implementation CKKSReloadAllItemsOperation

- (instancetype)init {
    return nil;
}

- (instancetype)initWithOperationDependencies:(CKKSOperationDependencies*)deps
{
    if(self = [super init]) {
        _deps = deps;
    }
    return self;
}

- (void)main {
#if TARGET_OS_TV
    [self.deps.personaAdapter prepareThreadForKeychainAPIUseForPersonaIdentifier: nil];
#endif
    id<CKKSDatabaseProviderProtocol> databaseProvider = self.deps.databaseProvider;

    for(CKKSKeychainViewState* viewState in self.deps.activeManagedViews) {
        [databaseProvider dispatchSyncWithSQLTransaction:^CKKSDatabaseTransactionResult{
            NSError* error = nil;

            NSArray<CKKSMirrorEntry*>* mirrorItems = [CKKSMirrorEntry allWithContextID:self.deps.contextID zoneID:viewState.zoneID error:&error];

            if(error) {
                ckkserror("ckksresync", viewState.zoneID, "Couldn't fetch mirror items: %@", error);
                self.error = error;
                return CKKSDatabaseTransactionRollback;
            }

            // Reload all entries back into the local keychain
            // We _could_ scan for entries, but that'd be expensive
            // In 36044942, we used to store only the CKRecord system fields in the ckrecord. To work around this, make a whole new CKRecord from the item.
            for(CKKSMirrorEntry* ckme in mirrorItems) {
                CKRecord* ckmeRecord = [ckme.item CKRecordWithZoneID:viewState.zoneID];
                if(!ckmeRecord) {
                    ckkserror("ckksresync", viewState.zoneID, "Couldn't make CKRecord for item: %@", ckme);
                    continue;
                }

                [self.deps intransactionCKRecordChanged:ckmeRecord resync:true];
            }

            return CKKSDatabaseTransactionCommit;
        }];
    }
}
@end
#endif

