/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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
#ifndef CloudKitDependencies_h
#define CloudKitDependencies_h

#import <ApplePushService/ApplePushService.h>
#import <CloudKit/CloudKit.h>
#import <Foundation/Foundation.h>
#import <Foundation/NSDistributedNotificationCenter.h>

NS_ASSUME_NONNULL_BEGIN

/* CKModifyRecordZonesOperation */
@protocol CKKSModifyRecordZonesOperation <NSObject>
+ (instancetype)alloc;
- (instancetype)initWithRecordZonesToSave:(nullable NSArray<CKRecordZone*>*)recordZonesToSave
                    recordZoneIDsToDelete:(nullable NSArray<CKRecordZoneID*>*)recordZoneIDsToDelete;

@property (nonatomic, strong, nullable) CKDatabase* database;
@property (nonatomic, copy, nullable) NSArray<CKRecordZone*>* recordZonesToSave;
@property (nonatomic, copy, nullable) NSArray<CKRecordZoneID*>* recordZoneIDsToDelete;
@property NSOperationQueuePriority queuePriority;
@property NSQualityOfService qualityOfService;
@property (nonatomic, strong, nullable) CKOperationGroup* group;
@property (nonatomic, copy, null_resettable) CKOperationConfiguration *configuration;

@property (nonatomic, copy, nullable) void (^modifyRecordZonesCompletionBlock)
    (NSArray<CKRecordZone*>* _Nullable savedRecordZones, NSArray<CKRecordZoneID*>* _Nullable deletedRecordZoneIDs, NSError* _Nullable operationError);

@end

@interface CKModifyRecordZonesOperation (SecCKKSModifyRecordZonesOperation) <CKKSModifyRecordZonesOperation>
;
@end

/* CKModifySubscriptionsOperation */
@protocol CKKSModifySubscriptionsOperation <NSObject>
+ (instancetype)alloc;
- (instancetype)initWithSubscriptionsToSave:(nullable NSArray<CKSubscription*>*)subscriptionsToSave
                    subscriptionIDsToDelete:(nullable NSArray<NSString*>*)subscriptionIDsToDelete;

@property (nonatomic, strong, nullable) CKDatabase* database;
@property (nonatomic, copy, nullable) NSArray<CKSubscription*>* subscriptionsToSave;
@property (nonatomic, copy, nullable) NSArray<NSString*>* subscriptionIDsToDelete;
@property NSOperationQueuePriority queuePriority;
@property NSQualityOfService qualityOfService;
@property (nonatomic, strong, nullable) CKOperationGroup* group;
@property (nonatomic, copy, null_resettable) CKOperationConfiguration *configuration;

@property (nonatomic, copy, nullable) void (^modifySubscriptionsCompletionBlock)
    (NSArray<CKSubscription*>* _Nullable savedSubscriptions, NSArray<NSString*>* _Nullable deletedSubscriptionIDs, NSError* _Nullable operationError);
@end

@interface CKModifySubscriptionsOperation (SecCKKSModifySubscriptionsOperation) <CKKSModifySubscriptionsOperation>
;
@end

/* CKFetchRecordZoneChangesOperation */
@protocol CKKSFetchRecordZoneChangesOperation <NSObject>
+ (instancetype)alloc;
- (instancetype)initWithRecordZoneIDs:(NSArray<CKRecordZoneID*>*)recordZoneIDs
                configurationsByRecordZoneID:(nullable NSDictionary<CKRecordZoneID*, CKFetchRecordZoneChangesConfiguration*>*)configurationsByRecordZoneID;

@property (nonatomic, strong, nullable) CKDatabase *database;
@property (nonatomic, copy, nullable) NSArray<CKRecordZoneID*>* recordZoneIDs;
@property (nonatomic, copy, nullable) NSDictionary<CKRecordZoneID*, CKFetchRecordZoneChangesConfiguration*>* configurationsByRecordZoneID;

@property (nonatomic, assign) BOOL fetchAllChanges;
@property (nonatomic, copy, nullable) void (^recordChangedBlock)(CKRecord* record);
@property (nonatomic, copy, nullable) void (^recordWithIDWasDeletedBlock)(CKRecordID* recordID, NSString* recordType);
@property (nonatomic, copy, nullable) void (^recordZoneChangeTokensUpdatedBlock)
    (CKRecordZoneID* recordZoneID, CKServerChangeToken* _Nullable serverChangeToken, NSData* _Nullable clientChangeTokenData);
@property (nonatomic, copy, nullable) void (^recordZoneFetchCompletionBlock)(CKRecordZoneID* recordZoneID,
                                                                             CKServerChangeToken* _Nullable serverChangeToken,
                                                                             NSData* _Nullable clientChangeTokenData,
                                                                             BOOL moreComing,
                                                                             NSError* _Nullable recordZoneError);
@property (nonatomic, copy, nullable) void (^fetchRecordZoneChangesCompletionBlock)(NSError* _Nullable operationError);

@property (nonatomic, strong, nullable) CKOperationGroup* group;
@property (nonatomic, copy, null_resettable) CKOperationConfiguration *configuration;

@property (nonatomic, copy) NSString *operationID;
@property (nonatomic, readonly, strong, nullable) CKOperationConfiguration *resolvedConfiguration;

@property (nonatomic, strong) NSString *deviceIdentifier;
@end

@interface CKFetchRecordZoneChangesOperation () <CKKSFetchRecordZoneChangesOperation>
@end

/* CKFetchRecordsOperation */
@protocol CKKSFetchRecordsOperation <NSObject>
+ (instancetype)alloc;
- (instancetype)init;
- (instancetype)initWithRecordIDs:(NSArray<CKRecordID*>*)recordIDs;

@property (nonatomic, strong, nullable) CKDatabase *database;
@property (nonatomic, copy, nullable) NSArray<CKRecordID*>* recordIDs;
@property (nonatomic, copy, nullable) NSArray<NSString*>* desiredKeys;
@property (nonatomic, strong, nullable) CKOperationGroup* group;
@property (nonatomic, copy, nullable) CKOperationConfiguration* configuration;
@property (nonatomic, copy, nullable) void (^perRecordProgressBlock)(CKRecordID* recordID, double progress);
@property (nonatomic, copy, nullable) void (^perRecordCompletionBlock)
    (CKRecord* _Nullable record, CKRecordID* _Nullable recordID, NSError* _Nullable error);
@property (nonatomic, copy, nullable) void (^fetchRecordsCompletionBlock)
    (NSDictionary<CKRecordID*, CKRecord*>* _Nullable recordsByRecordID, NSError* _Nullable operationError);
@end

@interface CKFetchRecordsOperation () <CKKSFetchRecordsOperation>
@end

/* CKQueryOperation */

@protocol CKKSQueryOperation <NSObject>
+ (instancetype)alloc;
- (instancetype)initWithQuery:(CKQuery*)query;
//Not implemented: - (instancetype)initWithCursor:(CKQueryCursor *)cursor;

@property (nonatomic, copy, nullable) CKQuery* query;
@property (nonatomic, copy, nullable) CKQueryCursor* cursor;

@property (nonatomic, copy, nullable) CKRecordZoneID* zoneID;
@property (nonatomic, assign) NSUInteger resultsLimit;
@property (nonatomic, copy, nullable) NSArray<NSString*>* desiredKeys;

@property (nonatomic, copy, nullable) void (^recordFetchedBlock)(CKRecord* record);
@property (nonatomic, copy, nullable) void (^queryCompletionBlock)(CKQueryCursor* _Nullable cursor, NSError* _Nullable operationError);
@end

@interface CKQueryOperation () <CKKSQueryOperation>
@end

/* APSConnection */
@protocol OctagonAPSConnection <NSObject>
@property NSArray<NSString*>* enabledTopics;
@property NSArray<NSString*>* opportunisticTopics;
@property NSArray<NSString*>* darkWakeTopics;

+ (instancetype)alloc;
- (id)initWithEnvironmentName:(NSString*)environmentName
            namedDelegatePort:(NSString*)namedDelegatePort
                        queue:(dispatch_queue_t)queue;


@property (nonatomic, readwrite, assign) id<APSConnectionDelegate> delegate;
@end

@interface APSConnection (SecOctagonAPSConnection) <OctagonAPSConnection>
@end

/* NSNotificationCenter */
@protocol CKKSNSNotificationCenter <NSObject>
+ (instancetype)defaultCenter;
- (void)addObserver:(id)observer selector:(SEL)aSelector name:(nullable NSNotificationName)aName object:(nullable id)anObject;
- (void)removeObserver:(id)observer;
@end
@interface NSNotificationCenter (CKKSMock) <CKKSNSNotificationCenter>
@end

@protocol CKKSNSDistributedNotificationCenter <NSObject>
+ (instancetype)defaultCenter;
- (void)addObserver:(id)observer selector:(SEL)aSelector name:(nullable NSNotificationName)aName object:(nullable id)anObject;
- (void)removeObserver:(id)observer;
- (void)postNotificationName:(NSNotificationName)name object:(nullable NSString *)object userInfo:(nullable NSDictionary *)userInfo options:(NSDistributedNotificationOptions)options;
@end

@interface NSDistributedNotificationCenter (CKKSMock) <CKKSNSDistributedNotificationCenter>
@end

/* Since CKDatabase doesn't share any types with NSOperationQueue, tell the type system about addOperation */
@protocol CKKSOperationQueue <NSObject>
- (void)addOperation:(NSOperation*)operation;
@end

@interface CKDatabase () <CKKSOperationQueue>
@end

@interface NSOperationQueue () <CKKSOperationQueue>
@end

NS_ASSUME_NONNULL_END

#endif /* CloudKitDependencies_h */
