/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
#if OCTAGON

#import "keychain/ckks/CKKSDeleteCurrentItemPointersOperation.h"

#import "keychain/ckks/CKKSCurrentItemPointer.h"
#import "keychain/ot/ObjCImprovements.h"

@interface CKKSDeleteCurrentItemPointersOperation ()
@property (nullable) CKModifyRecordsOperation* modifyRecordsOperation;
@property (nullable) CKOperationGroup* ckoperationGroup;

@property CKKSOperationDependencies* deps;

@property (nonnull) NSString* accessGroup;

@property (nonnull) NSArray<NSString*>* identifiers;
@end

@implementation CKKSDeleteCurrentItemPointersOperation

- (instancetype)initWithCKKSOperationDependencies:(CKKSOperationDependencies*)operationDependencies
                                        viewState:(CKKSKeychainViewState*)viewState
                                      accessGroup:(NSString*)accessGroup
                                      identifiers:(NSArray<NSString*>*)identifiers
                                 ckoperationGroup:(CKOperationGroup* _Nullable)ckoperationGroup
{
    if ((self = [super init])) {
        _deps = operationDependencies;
        _viewState = viewState;
        _accessGroup = accessGroup;
        _identifiers = identifiers;
        _ckoperationGroup = ckoperationGroup;
    }
    return self;
}

- (void)groupStart
{
    WEAKIFY(self);
#if TARGET_OS_TV
    [self.deps.personaAdapter prepareThreadForKeychainAPIUseForPersonaIdentifier: nil];
#endif
    [self.deps.databaseProvider dispatchSyncWithSQLTransaction:^CKKSDatabaseTransactionResult {
        if(self.cancelled) {
            ckksnotice("ckkscurrent", self.viewState.zoneID, "CKKSDeleteCurrentItemPointersOperation cancelled, quitting");
            return CKKSDatabaseTransactionRollback;
        }

        ckksnotice("ckkscurrent", self.viewState.zoneID, "Deleting current item pointers (%lu)", (unsigned long)self.identifiers.count);

        NSMutableArray<CKRecordID*>* recordIDsToDelete = [[NSMutableArray alloc] init];
        for (NSString* identifier in self.identifiers) {
            NSString* recordName = [NSString stringWithFormat:@"%@-%@", self.accessGroup, identifier];
            CKRecordID* recordID = [[CKRecordID alloc] initWithRecordName:recordName zoneID:self.viewState.zoneID];
            [recordIDsToDelete addObject:recordID];
        }

        // Start a CKModifyRecordsOperation to delete current item pointers
        NSBlockOperation* modifyComplete = [[NSBlockOperation alloc] init];
        modifyComplete.name = @"deleteCurrentItemPointers-modifyRecordsComplete";
        [self dependOnBeforeGroupFinished:modifyComplete];

        self.modifyRecordsOperation = [[CKModifyRecordsOperation alloc] initWithRecordsToSave:nil recordIDsToDelete:recordIDsToDelete];
        self.modifyRecordsOperation.atomic = YES;
        // We're likely rolling a PCS identity, or creating a new one. User cares.
        self.modifyRecordsOperation.configuration.isCloudKitSupportOperation = YES;

        // CKKSSetHighPriorityOperations is default enabled
        // This operation might be needed during CKKS/Manatee bringup, which affects the user experience. Bump our priority to get it off-device and unblock Manatee access.
        self.modifyRecordsOperation.qualityOfService = NSQualityOfServiceUserInitiated;
        

        self.modifyRecordsOperation.group = self.ckoperationGroup;

        self.modifyRecordsOperation.perRecordDeleteBlock = ^(CKRecordID* recordID, NSError* error) {
            STRONGIFY(self);

            if(!error) {
                ckksnotice("ckkscurrent", self.viewState.zoneID, "Current pointer delete successful for %@", recordID.recordName);
            } else {
                ckkserror("ckkscurrent", self.viewState.zoneID, "error on row: %@ %@", error, recordID);
            }
        };

        self.modifyRecordsOperation.modifyRecordsCompletionBlock = ^(NSArray<CKRecord *> *savedRecords, NSArray<CKRecordID *> *deletedRecordIDs, NSError *ckerror) {
            STRONGIFY(self);
            id<CKKSDatabaseProviderProtocol> databaseProvider = self.deps.databaseProvider;

            if(ckerror) {
                ckkserror("ckkscurrent", self.viewState.zoneID, "CloudKit returned an error: %@", ckerror);
                self.error = ckerror;

                [self.operationQueue addOperation:modifyComplete];
                return;
            }

            __block NSError* error = nil;

            [databaseProvider dispatchSyncWithSQLTransaction:^CKKSDatabaseTransactionResult{
                for(CKRecordID* recordID in deletedRecordIDs) {
                    if(![CKKSCurrentItemPointer intransactionRecordDeleted:recordID contextID:self.deps.contextID resync:false error:&error]) {
                        ckkserror("ckkscurrent", self.viewState.zoneID, "Couldn't delete current item pointer for %@ from database: %@", recordID.recordName, error);
                        self.error = error;
                    }

                    // Schedule a 'view changed' notification
                    [self.viewState.notifyViewChangedScheduler trigger];
                }
                return CKKSDatabaseTransactionCommit;
            }];

            self.error = error;
            [self.operationQueue addOperation:modifyComplete];
        };

        [self dependOnBeforeGroupFinished: self.modifyRecordsOperation];
        [self.deps.ckdatabase addOperation:self.modifyRecordsOperation];

        return CKKSDatabaseTransactionCommit;
    }];
}

@end

#endif // OCTAGON
