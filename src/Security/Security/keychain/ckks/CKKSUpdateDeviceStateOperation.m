/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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

#include <utilities/SecInternalReleasePriv.h>
#import "keychain/ckks/CKKSKeychainView.h"
#import "keychain/ckks/CKKSUpdateDeviceStateOperation.h"
#import "keychain/ckks/CKKSCurrentKeyPointer.h"
#import "keychain/ckks/CKKSKey.h"
#import "keychain/ckks/CKKSLockStateTracker.h"
#import "keychain/ckks/CKKSSQLDatabaseObject.h"
#import "keychain/ot/ObjCImprovements.h"

@interface CKKSUpdateDeviceStateOperation ()
@property CKOperationGroup* group;
@property bool rateLimit;
@end

@implementation CKKSUpdateDeviceStateOperation

- (instancetype)initWithOperationDependencies:(CKKSOperationDependencies*)operationDependencies
                                    rateLimit:(bool)rateLimit
                             ckoperationGroup:(CKOperationGroup*)group
{
    if((self = [super init])) {
        _deps = operationDependencies;
        _group = group;
        _rateLimit = rateLimit;
    }
    return self;
}

- (void)groupStart {
#if TARGET_OS_TV
    [self.deps.personaAdapter prepareThreadForKeychainAPIUseForPersonaIdentifier: nil];
#endif
    CKKSAccountStateTracker* accountTracker = self.deps.accountStateTracker;
    if(!accountTracker) {
        ckkserror_global("ckksdevice", "no AccountTracker object");
        self.error = [NSError errorWithDomain:CKKSErrorDomain code:CKKSErrorUnexpectedNil userInfo:@{NSLocalizedDescriptionKey: @"no AccountTracker object"}];
        return;
    }

    WEAKIFY(self);

    // We must have the ck device ID to run this operation.
    if([accountTracker.ckdeviceIDInitialized wait:200*NSEC_PER_SEC]) {
        ckkserror_global("ckksdevice", "CK device ID not initialized, likely quitting");
    }

    NSString* ckdeviceID = accountTracker.ckdeviceID;
    if(!ckdeviceID) {
        ckkserror_global("ckksdevice", "CK device ID not initialized, quitting");
        self.error = [NSError errorWithDomain:CKKSErrorDomain
                                         code:CKKSNoCloudKitDeviceID
                                     userInfo:@{NSLocalizedDescriptionKey: @"CK device ID missing", NSUnderlyingErrorKey:CKKSNilToNSNull(accountTracker.ckdeviceIDError)}];
        return;
    }

    // We'd also really like to know the HSA2/Managed-ness of the world
    if([accountTracker.cdpCapableiCloudAccountInitialized wait:500*NSEC_PER_MSEC]) {
        ckkserror_global("ckksdevice", "Not quite sure if the account is HSA2/Managed or not. Probably will quit?");
    }

    NSHashTable<CKDatabaseOperation*>* ckOperations = [NSHashTable weakObjectsHashTable];

    id<CKKSDatabaseProviderProtocol> databaseProvider = self.deps.databaseProvider;

    for(CKKSKeychainViewState* viewState in self.deps.activeManagedViews) {
        [databaseProvider dispatchSyncWithSQLTransaction:^CKKSDatabaseTransactionResult{
            NSError* error = nil;

            CKKSDeviceStateEntry* cdse = [CKKSDeviceStateEntry intransactionCreateDeviceStateForView:viewState
                                                                                      accountTracker:self.deps.accountStateTracker
                                                                                    lockStateTracker:self.deps.lockStateTracker
                                                                                               error:&error];
            if(error || !cdse) {
                ckkserror("ckksdevice", viewState.zoneID, "Error creating device state entry; quitting: %@", error);
                self.error = error;
                return CKKSDatabaseTransactionRollback;
            }

            if(self.rateLimit) {
                NSDate* lastUpdate = cdse.storedCKRecord.modificationDate;

                // Only upload this every 3 days (1 day for internal installs)
                NSDate* now = [NSDate date];
                NSDateComponents* offset = [[NSDateComponents alloc] init];
                if(SecIsInternalRelease()) {
                    [offset setHour:-23];
                } else {
                    [offset setHour:-3*24];
                }
                NSDate* deadline = [[NSCalendar currentCalendar] dateByAddingComponents:offset toDate:now options:0];

                if(lastUpdate == nil || [lastUpdate compare: deadline] == NSOrderedAscending) {
                    ckksnotice("ckksdevice", viewState.zoneID, "Not rate-limiting: last updated %@ vs %@", lastUpdate, deadline);
                } else {
                    ckksnotice("ckksdevice", viewState.zoneID, "Last update is within 3 days (%@); rate-limiting this operation", lastUpdate);
                    self.error =  [NSError errorWithDomain:CKKSErrorDomain
                                                      code:CKKSErrorRateLimited
                                                  userInfo:@{NSLocalizedDescriptionKey: @"Rate-limited the CKKSUpdateDeviceStateOperation"}];
                    return CKKSDatabaseTransactionRollback;
                }
            }

            ckksnotice("ckksdevice", viewState.zoneID, "Saving new device state %@", cdse);

            NSArray* recordsToSave = @[[cdse CKRecordWithZoneID:viewState.zoneID]];

            // Start a CKModifyRecordsOperation to save this new/updated record.
            NSBlockOperation* modifyComplete = [[NSBlockOperation alloc] init];
            modifyComplete.name = @"updateDeviceState-modifyRecordsComplete";
            [self dependOnBeforeGroupFinished: modifyComplete];

            CKModifyRecordsOperation* zoneModifyRecordsOperation = [[CKModifyRecordsOperation alloc] initWithRecordsToSave:recordsToSave recordIDsToDelete:nil];
            zoneModifyRecordsOperation.atomic = TRUE;
            zoneModifyRecordsOperation.qualityOfService = NSQualityOfServiceUtility;
            zoneModifyRecordsOperation.savePolicy = CKRecordSaveAllKeys; // Overwrite anything in CloudKit: this is our state now
            zoneModifyRecordsOperation.group = self.group;

            zoneModifyRecordsOperation.perRecordSaveBlock = ^(CKRecordID *recordID, CKRecord * _Nullable record, NSError * _Nullable error) {
                if(!error) {
                    ckksnotice("ckksdevice", viewState.zoneID, "Device state record upload successful for %@: %@", recordID.recordName, record);
                } else {
                    ckkserror("ckksdevice", viewState.zoneID, "error on row: %@ %@", error, recordID);
                }
            };

            zoneModifyRecordsOperation.modifyRecordsCompletionBlock = ^(NSArray<CKRecord *> *savedRecords, NSArray<CKRecordID *> *deletedRecordIDs, NSError *ckerror) {
                STRONGIFY(self);

                if(ckerror) {
                    ckkserror("ckksdevice", viewState.zoneID, "CloudKit returned an error: %@", ckerror);
                    self.error = ckerror;
                    [self runBeforeGroupFinished:modifyComplete];
                    return;
                }

                __block NSError* error = nil;

                [self.deps.databaseProvider dispatchSyncWithSQLTransaction:^CKKSDatabaseTransactionResult{
                    for(CKRecord* record in savedRecords) {
                        // Save the item records
                        if([record.recordType isEqualToString: SecCKRecordDeviceStateType]) {
                            CKKSDeviceStateEntry* newcdse = [[CKKSDeviceStateEntry alloc] initWithCKRecord:record contextID:self.deps.contextID];
                            [newcdse saveToDatabase:&error];
                            if(error) {
                                ckkserror("ckksdevice", viewState.zoneID, "Couldn't save new device state(%@) to database: %@", newcdse, error);
                            }
                        }
                    }
                    return CKKSDatabaseTransactionCommit;
                }];

                self.error = error;
                [self runBeforeGroupFinished:modifyComplete];
            };

            [zoneModifyRecordsOperation linearDependencies:ckOperations];
            [self dependOnBeforeGroupFinished:zoneModifyRecordsOperation];
            [self.deps.ckdatabase addOperation:zoneModifyRecordsOperation];

            return CKKSDatabaseTransactionCommit;
        }];
    }
}

@end

#endif // OCTAGON
