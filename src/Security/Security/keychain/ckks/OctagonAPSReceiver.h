/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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
#import <ApplePushService/ApplePushService.h>
#import <CloudKit/CloudKit.h>
#import "keychain/ckks/CKKSCondition.h"
#import "keychain/ckks/CloudKitDependencies.h"


NS_ASSUME_NONNULL_BEGIN

// APS is giving us a tracingUUID and a tracingEnabled bool, but our interfaces take a CKRecordZoneNotification. Add them to that class, then.
@interface CKRecordZoneNotification (CKKSPushTracing)
@property (nonatomic, assign) BOOL ckksPushTracingEnabled;
@property (nonatomic, strong, nullable) NSString* ckksPushTracingUUID;
@property (nonatomic, strong, nullable) NSDate* ckksPushReceivedDate;
@end

@protocol CKKSZoneUpdateReceiverProtocol <NSObject>
- (void)notifyZoneChange:(CKRecordZoneNotification* _Nullable)notification;
@end

@protocol OctagonCuttlefishUpdateReceiver <NSObject>
- (void)notifyContainerChange:(APSIncomingMessage* _Nullable)notification;
@end

@interface OctagonAPSReceiver : NSObject <APSConnectionDelegate>

// class dependencies (for injection)
@property (readonly) Class<OctagonAPSConnection> apsConnectionClass;
@property (nullable) id<OctagonAPSConnection> apsConnection;

@property (readonly) BOOL haveStalePushes;

+ (instancetype)receiverForNamedDelegatePort:(NSString*)namedDelegatePort
                          apsConnectionClass:(Class<OctagonAPSConnection>)apsConnectionClass;
+ (void)resetGlobalDelegatePortMap;

- (void)registerForEnvironment:(NSString*)environmentName;

- (CKKSCondition*)registerCKKSReceiver:(id<CKKSZoneUpdateReceiverProtocol>)receiver
                             contextID:(NSString*)contextID;

// APS reserves the right to coalesce pushes by topic. So, any cuttlefish container push might hide pushes for other cuttlefish containers.
// This is okay for now, as we only have one active cuttlefish container per device, but if we start to have multiple accounts, this handling might need to change.
- (CKKSCondition*)registerCuttlefishReceiver:(id<OctagonCuttlefishUpdateReceiver>)receiver
                            forContainerName:(NSString*)containerName
                                   contextID:(NSString*)contextID;
// Test support:
- (instancetype)initWithNamedDelegatePort:(NSString*)namedDelegatePort
                     apsConnectionClass:(Class<OctagonAPSConnection>)apsConnectionClass;
- (instancetype)initWithNamedDelegatePort:(NSString*)namedDelegatePort
                     apsConnectionClass:(Class<OctagonAPSConnection>)apsConnectionClass
                       stalePushTimeout:(uint64_t)stalePushTimeout;

// This is the queue that APNS will use send the notifications to us
+ (dispatch_queue_t)apsDeliveryQueue;
- (NSArray<NSString *>*)registeredPushEnvironments;
@end

@interface OctagonAPSReceiver (Testing)
- (void)reportDroppedPushes:(NSSet<CKRecordZoneNotification*>*)notifications;
@end

NS_ASSUME_NONNULL_END
#endif  // OCTAGON
