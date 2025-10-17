/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 10, 2022.
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
#ifndef IOHIDSensorPowerLoggingFilter_h
#define IOHIDSensorPowerLoggingFilter_h

#import <HID/HID_Private.h>

NS_ASSUME_NONNULL_BEGIN

@interface IOHIDSensorPowerLoggingFilter : NSObject <HIDServiceFilter>

+ (BOOL)matchService:(HIDEventService *)service
             options:(nullable NSDictionary *)options
               score:(NSInteger *)score;

- (nullable instancetype)initWithService:(HIDEventService *)service;

- (nullable id)propertyForKey:(NSString *)key
                       client:(nullable HIDConnection *)client;

- (BOOL)setProperty:(nullable id)value
             forKey:(NSString *)key
             client:(nullable HIDConnection *)client;

- (nullable HIDEvent *)filterEvent:(HIDEvent *)event;

- (nullable HIDEvent *)filterEventMatching:(nullable NSDictionary *)matching
                                     event:(HIDEvent *)event
                                 forClient:(nullable HIDConnection *)client;

- (void)setCancelHandler:(HIDBlock)handler;

- (void)activate;

- (void)cancel;

- (void)clientNotification:(HIDConnection *)client added:(BOOL)added;

// should be weak
@property (weak) HIDEventService        * service;

@end

NS_ASSUME_NONNULL_END

#endif /* IOHIDSensorPowerLoggingFilter_h */
