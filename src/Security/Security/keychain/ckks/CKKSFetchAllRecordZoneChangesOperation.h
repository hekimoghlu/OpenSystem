/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 13, 2024.
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
#import <Foundation/Foundation.h>

#if OCTAGON
@class CKKSKeychainView;
#import "keychain/ckks/CKKSGroupOperation.h"
#import "keychain/ckks/CloudKitDependencies.h"

NS_ASSUME_NONNULL_BEGIN

/* Fetch Reasons */
@protocol CKKSFetchBecauseProtocol <NSObject>
@end
typedef NSString<CKKSFetchBecauseProtocol> CKKSFetchBecause;
extern CKKSFetchBecause* const CKKSFetchBecauseAPNS;
extern CKKSFetchBecause* const CKKSFetchBecauseAPIFetchRequest;
extern CKKSFetchBecause* const CKKSFetchBecauseSEAPIFetchRequest;
extern CKKSFetchBecause* const CKKSFetchBecauseKeySetFetchRequest;
extern CKKSFetchBecause* const CKKSFetchBecauseCurrentItemFetchRequest;
extern CKKSFetchBecause* const CKKSFetchBecauseInitialStart;
extern CKKSFetchBecause* const CKKSFetchBecausePreviousFetchFailed;
extern CKKSFetchBecause* const CKKSFetchBecauseKeyHierarchy;
extern CKKSFetchBecause* const CKKSFetchBecauseTesting;
extern CKKSFetchBecause* const CKKSFetchBecauseResync;
extern CKKSFetchBecause* const CKKSFetchBecauseMoreComing;
extern CKKSFetchBecause* const CKKSFetchBecauseResolvingConflict;
extern CKKSFetchBecause* const CKKSFetchBecausePeriodicRefetch;
extern CKKSFetchBecause* const CKKSFetchBecauseOctagonPairingComplete;

/* Clients that register to use fetches */
@interface CKKSCloudKitFetchRequest : NSObject
@property bool participateInFetch;

// If true, you will receive YES in the resync parameter to your callback.
// You may also receive YES to your callback if a resync has been triggered for you.
// It does nothing else. Use as you see fit.
// Note: you will receive exactly one callback with moreComing=0 and resync=1 for each
// resync fetch. You may then receive further callbacks with resync=0 during the same fetch,
// if other clients keep needing fetches.
@property BOOL resync;

@property (nullable) CKServerChangeToken* changeToken;
@end

@class CKKSCloudKitDeletion;

@protocol CKKSChangeFetcherClient <NSObject>
- (BOOL)zoneIsReadyForFetching:(CKRecordZoneID*)zoneID;
- (CKKSCloudKitFetchRequest*)participateInFetch:(CKRecordZoneID*)zoneID;

// Return false if this is a 'fatal' error and you don't want another fetch to be tried
- (bool)shouldRetryAfterFetchError:(NSError*)error
                            zoneID:(CKRecordZoneID*)zoneID;

- (void)changesFetched:(NSArray<CKRecord*>*)changedRecords
      deletedRecordIDs:(NSArray<CKKSCloudKitDeletion*>*)deleted
                zoneID:(CKRecordZoneID*)zoneID
        newChangeToken:(CKServerChangeToken*)changeToken
            moreComing:(BOOL)moreComing
                resync:(BOOL)resync;
@end

// I don't understand why recordType isn't part of record ID, but deletions come in as both things
@interface CKKSCloudKitDeletion : NSObject
@property CKRecordID* recordID;
@property NSString* recordType;
- (instancetype)initWithRecordID:(CKRecordID*)recordID recordType:(NSString*)recordType;
@end


@interface CKKSFetchAllRecordZoneChangesOperation : CKKSGroupOperation
@property (readonly) Class<CKKSFetchRecordZoneChangesOperation> fetchRecordZoneChangesOperationClass;
@property (readonly) CKContainer* container;

// Set this to true before starting this operation if you'd like resync behavior:
//  Fetching everything currently in CloudKit and comparing to local copy
@property bool resync;

@property (nullable) NSMutableArray<CKRecordZoneID*>* fetchedZoneIDs;

@property NSSet<CKKSFetchBecause*>* fetchReasons;
@property NSSet<CKRecordZoneNotification*>* apnsPushes;

@property NSMutableDictionary<CKRecordID*, CKRecord*>* modifications;
@property NSMutableDictionary<CKRecordID*, CKKSCloudKitDeletion*>* deletions;
@property NSMutableDictionary<CKRecordZoneID*, CKServerChangeToken*>* changeTokens;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithContainer:(CKContainer*)container
                       fetchClass:(Class<CKKSFetchRecordZoneChangesOperation>)fetchRecordZoneChangesOperationClass
                        clientMap:(NSDictionary<CKRecordZoneID*, id<CKKSChangeFetcherClient>>*)clientMap
                     fetchReasons:(NSSet<CKKSFetchBecause*>*)fetchReasons
                       apnsPushes:(NSSet<CKRecordZoneNotification*>* _Nullable)apnsPushes
                      forceResync:(bool)forceResync
                 ckoperationGroup:(CKOperationGroup*)ckoperationGroup
                          altDSID:(NSString*)altDSID
                       sendMetric:(bool)sendMetric;

@end

NS_ASSUME_NONNULL_END

#endif
