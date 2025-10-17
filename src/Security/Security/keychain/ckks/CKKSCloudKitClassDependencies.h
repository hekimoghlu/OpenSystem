/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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

#import "keychain/ckks/CKKSNotifier.h"
#import "keychain/ckks/CloudKitDependencies.h"

NS_ASSUME_NONNULL_BEGIN

@interface CKKSCloudKitClassDependencies : NSObject
@property (readonly) Class<CKKSFetchRecordZoneChangesOperation> fetchRecordZoneChangesOperationClass;
@property (readonly) Class<CKKSFetchRecordsOperation> fetchRecordsOperationClass;
@property (readonly) Class<CKKSQueryOperation> queryOperationClass;
@property (readonly) Class<CKKSModifySubscriptionsOperation> modifySubscriptionsOperationClass;
@property (readonly) Class<CKKSModifyRecordZonesOperation> modifyRecordZonesOperationClass;
@property (readonly) Class<OctagonAPSConnection> apsConnectionClass;
@property (readonly) Class<CKKSNSNotificationCenter> nsnotificationCenterClass;
@property (readonly) Class<CKKSNSDistributedNotificationCenter> nsdistributednotificationCenterClass;
@property (readonly) Class<CKKSNotifier> notifierClass;

- (instancetype)init NS_UNAVAILABLE;

+ (CKKSCloudKitClassDependencies*) forLiveCloudKit;

- (instancetype)initWithFetchRecordZoneChangesOperationClass:(Class<CKKSFetchRecordZoneChangesOperation>)fetchRecordZoneChangesOperationClass
                                  fetchRecordsOperationClass:(Class<CKKSFetchRecordsOperation>)fetchRecordsOperationClass
                                         queryOperationClass:(Class<CKKSQueryOperation>)queryOperationClass
                           modifySubscriptionsOperationClass:(Class<CKKSModifySubscriptionsOperation>)modifySubscriptionsOperationClass
                             modifyRecordZonesOperationClass:(Class<CKKSModifyRecordZonesOperation>)modifyRecordZonesOperationClass
                                          apsConnectionClass:(Class<OctagonAPSConnection>)apsConnectionClass
                                   nsnotificationCenterClass:(Class<CKKSNSNotificationCenter>)nsnotificationCenterClass
                        nsdistributednotificationCenterClass:(Class<CKKSNSDistributedNotificationCenter>)nsdistributednotificationCenterClass
                                               notifierClass:(Class<CKKSNotifier>)notifierClass;
@end


NS_ASSUME_NONNULL_END

#endif  // Octagon
