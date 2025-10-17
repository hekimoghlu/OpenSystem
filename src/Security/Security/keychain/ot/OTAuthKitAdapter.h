/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 25, 2024.
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

#import <Foundation/Foundation.h>
#import "keychain/ot/OTDefines.h"
#import <AuthKit/AuthKit.h>
#import <AuthKit/AuthKit_Private.h>

NS_ASSUME_NONNULL_BEGIN

@protocol OTAuthKitAdapterNotifier
- (void)notificationOfMachineIDListChange;
@end

@protocol OTAuthKitAdapter

- (BOOL)accountIsCDPCapableByAltDSID:(NSString*)altDSID;
- (BOOL)accountIsDemoAccountByAltDSID:(NSString*)altDSID error:(NSError**)error NS_SWIFT_NOTHROW;

- (NSString* _Nullable)machineID:(NSString* _Nullable)altDSID
                          flowID:(NSString* _Nullable)flowID
                 deviceSessionID:(NSString* _Nullable)deviceSessionID
                  canSendMetrics:(BOOL)canSendMetrics
                           error:(NSError**)error;

- (void)fetchCurrentDeviceListByAltDSID:(NSString*)altDSID 
                                 flowID:(NSString*)flowID
                        deviceSessionID:(NSString*)deviceSessionID
                                  reply:(void (^)(NSSet<NSString*>* _Nullable machineIDs,
                                                  NSSet<NSString*>* _Nullable userInitiatedRemovals,
                                                  NSSet<NSString*>* _Nullable evictedRemovals,
                                                  NSSet<NSString*>* _Nullable unknownReasonRemovals,
                                                  NSString* _Nullable version,
                                                  NSString* _Nullable trustedDeviceHash,
                                                  NSString* _Nullable deletedDeviceHash,
                                                  NSNumber* _Nullable trustedDevicesUpdateTimestamp,
                                                  NSError* _Nullable error))complete;

- (void)registerNotification:(id<OTAuthKitAdapterNotifier>)notifier;

@end

@interface OTAuthKitActualAdapter : NSObject <OTAuthKitAdapter>
@end

NS_ASSUME_NONNULL_END

#endif // OCTAGON

