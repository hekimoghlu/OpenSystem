/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 12, 2022.
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
#import <CloudKit/CloudKit.h>
#import <Foundation/Foundation.h>
#import "keychain/ckks/CKKSResultOperation.h"
#import "keychain/ckks/CKKSFetchAllRecordZoneChangesOperation.h"
#import "keychain/ckks/OctagonAPSReceiver.h"

NS_ASSUME_NONNULL_BEGIN

/*
 * This class implements a CloudKit-fetch-with-retry.
 * In the case of network or other failures, it'll issue retries.
 * Only in the case of a clean fetch will its operation dependency resolve.
 */

@class CKKSKeychainView;
@class CKKSReachabilityTracker;
@class CKKSNearFutureScheduler;

@interface CKKSZoneChangeFetcher : NSObject <CKKSZoneUpdateReceiverProtocol>
@property (readonly) Class<CKKSFetchRecordZoneChangesOperation> fetchRecordZoneChangesOperationClass;
@property (readonly) CKContainer* container;
@property CKKSReachabilityTracker* reachabilityTracker;

@property (readonly) NSError* lastCKFetchError;

@property bool sendMetric;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithContainer:(CKContainer*)container
                       fetchClass:(Class<CKKSFetchRecordZoneChangesOperation>)fetchRecordZoneChangesOperationClass
              reachabilityTracker:(CKKSReachabilityTracker *)reachabilityTracker
                          altDSID:(NSString*)altDSID
                       sendMetric:(bool)sendMetric;

- (void)registerClient:(id<CKKSChangeFetcherClient>)client zoneID:(CKRecordZoneID*)zoneID;

- (CKKSResultOperation*)requestSuccessfulFetch:(CKKSFetchBecause*)why;
- (CKKSResultOperation*)requestSuccessfulFetchForManyReasons:(NSSet<CKKSFetchBecause*>*)why;

// Returns the next fetch, if one is scheduled, or the last/currently executing fetch if not.
- (CKKSResultOperation* _Nullable)inflightFetch;

// CKKSZoneUpdateReceiverProtocol
- (void)notifyZoneChange:(CKRecordZoneNotification* _Nullable)notification;

// We don't particularly care what this does, as long as it finishes
- (void)holdFetchesUntil:(CKKSResultOperation* _Nullable)holdOperation;

- (void)cancel;
- (void)halt;

// I don't recommend using these unless you're a test.
@property CKKSNearFutureScheduler* fetchScheduler;
@end

NS_ASSUME_NONNULL_END
#endif
