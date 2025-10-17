/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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

NS_ASSUME_NONNULL_BEGIN

@interface AAFAnalyticsEventSecurity : NSObject

@property (retain) dispatch_queue_t queue;

- (instancetype)initWithKeychainCircleMetrics:(NSDictionary * _Nullable)metrics
                                      altDSID:(NSString * _Nullable)altDSID
                                       flowID:(NSString * _Nullable)flowID
                              deviceSessionID:(NSString * _Nullable)deviceSessionID
                                    eventName:(NSString *)eventName
                              testsAreEnabled:(BOOL)testsAreEnabled
                               canSendMetrics:(BOOL)canSendMetrics
                                     category:(NSNumber *)category;

- (instancetype)initWithCKKSMetrics:(NSDictionary * _Nullable)metrics
                            altDSID:(NSString *)altDSID
                          eventName:(NSString *)eventName
                    testsAreEnabled:(BOOL)testsAreEnabled
                           category:(NSNumber *)category
                         sendMetric:(BOOL)sendMetric;

- (instancetype)initWithKeychainCircleMetrics:(NSDictionary * _Nullable)metrics
                                      altDSID:(NSString * _Nullable)altDSID
                                    eventName:(NSString *)eventName
                                     category:(NSNumber *)category;
- (id)getEvent;
- (void)addMetrics:(NSDictionary*)metrics;
- (void)populateUnderlyingErrorsStartingWithRootError:(NSError* _Nullable)error;
- (BOOL)permittedToSendMetrics;

#if __has_include(<AAAFoundation/AAAFoundation.h>)
+ (NSString* _Nullable)fetchDeviceSessionIDFromAuthKit:(NSString*)altDSID;
#endif

@end

NS_ASSUME_NONNULL_END
