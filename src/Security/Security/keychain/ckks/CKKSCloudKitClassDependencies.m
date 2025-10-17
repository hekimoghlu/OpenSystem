/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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


#import "keychain/ckks/CKKSCloudKitClassDependencies.h"

@implementation CKKSCloudKitClassDependencies

+ (CKKSCloudKitClassDependencies*) forLiveCloudKit
{
    return [[CKKSCloudKitClassDependencies alloc] initWithFetchRecordZoneChangesOperationClass:[CKFetchRecordZoneChangesOperation class]
                                                                    fetchRecordsOperationClass:[CKFetchRecordsOperation class]
                                                                           queryOperationClass:[CKQueryOperation class]
                                                             modifySubscriptionsOperationClass:[CKModifySubscriptionsOperation class]
                                                               modifyRecordZonesOperationClass:[CKModifyRecordZonesOperation class]
                                                                            apsConnectionClass:[APSConnection class]
                                                                     nsnotificationCenterClass:[NSNotificationCenter class]
                                                          nsdistributednotificationCenterClass:[NSDistributedNotificationCenter class]
                                                                                 notifierClass:[CKKSNotifyPostNotifier class]];
}

- (instancetype)initWithFetchRecordZoneChangesOperationClass:(Class<CKKSFetchRecordZoneChangesOperation>)fetchRecordZoneChangesOperationClass
                                  fetchRecordsOperationClass:(Class<CKKSFetchRecordsOperation>)fetchRecordsOperationClass
                                         queryOperationClass:(Class<CKKSQueryOperation>)queryOperationClass
                           modifySubscriptionsOperationClass:(Class<CKKSModifySubscriptionsOperation>)modifySubscriptionsOperationClass
                             modifyRecordZonesOperationClass:(Class<CKKSModifyRecordZonesOperation>)modifyRecordZonesOperationClass
                                          apsConnectionClass:(Class<OctagonAPSConnection>)apsConnectionClass
                                   nsnotificationCenterClass:(Class<CKKSNSNotificationCenter>)nsnotificationCenterClass
                        nsdistributednotificationCenterClass:(Class<CKKSNSDistributedNotificationCenter>)nsdistributednotificationCenterClass
                                               notifierClass:(Class<CKKSNotifier>)notifierClass
{
    if(self = [super init]) {
        _fetchRecordZoneChangesOperationClass = fetchRecordZoneChangesOperationClass;
        _fetchRecordsOperationClass = fetchRecordsOperationClass;
        _queryOperationClass = queryOperationClass;
        _modifySubscriptionsOperationClass = modifySubscriptionsOperationClass;
        _modifyRecordZonesOperationClass = modifyRecordZonesOperationClass;
        _apsConnectionClass = apsConnectionClass;
        _nsnotificationCenterClass = nsnotificationCenterClass;
        _nsdistributednotificationCenterClass = nsdistributednotificationCenterClass;
        _notifierClass = notifierClass;
    }
    return self;
}

@end

