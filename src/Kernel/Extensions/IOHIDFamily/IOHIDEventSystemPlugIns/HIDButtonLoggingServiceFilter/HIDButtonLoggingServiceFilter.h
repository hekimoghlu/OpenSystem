/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 17, 2024.
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
#pragma once

#import <HID/HID_Private.h>

os_log_t _IOHIDButtonLog(void);

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, HIDSuppressionStateType) {
    kHIDSuppressionStateTypeUnknown = 0,
    kHIDSuppressionStateTypeSuppressed = 1,
    kHIDSuppressionStateTypeUnsuppressed = 2,
};

@interface HIDButtonLoggingServiceFilter: NSObject <HIDServiceFilter>

- (nullable instancetype)initWithService:(HIDEventService *)service;

- (nullable id)propertyForKey:(NSString *)key
                       client:(nullable HIDConnection *)client;

- (BOOL)setProperty:(nullable id)value
             forKey:(NSString *)key
             client:(nullable HIDConnection *)client;

+ (BOOL)matchService:(HIDEventService *)service
             options:(nullable NSDictionary *)options
               score:(NSInteger *)score;

- (nullable HIDEvent *)filterEvent:(HIDEvent *)event;

- (nullable HIDEvent *)filterEventMatching:(nullable NSDictionary *)matching
                                     event:(HIDEvent *)event
                                 forClient:(nullable HIDConnection *)client;

- (void)setCancelHandler:(HIDBlock)handler;

- (void)activate;

- (void)cancel;

// should be weak
@property (weak) HIDEventService        *service;

@end

NS_ASSUME_NONNULL_END
