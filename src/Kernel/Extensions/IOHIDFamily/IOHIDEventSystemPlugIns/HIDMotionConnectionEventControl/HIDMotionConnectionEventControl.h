/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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
#ifndef HIDMotionConnectionEventControl_h
#define HIDMotionConnectionEventControl_h

#import <HID/HIDBase.h>
#import <HID/HID_Private.h>

NS_ASSUME_NONNULL_BEGIN

@interface HIDMotionConnectionEventControl : NSObject <HIDConnectionPlugin>

- (nullable instancetype)initWithConnection:(HIDConnection *)connection;

- (nullable id)propertyForKey:(NSString *)key;

- (BOOL)setProperty:(nullable id)value forKey:(NSString *)key;

+ (BOOL)matchConnection:(HIDConnection *)connection;

- (nullable HIDEvent *)filterEvent:(HIDEvent *)event;

- (void)setCancelHandler:(HIDBlock)handler;

- (void)activate;

- (void)cancel;

- (void)setDispatchQueue:(dispatch_queue_t)queue;

@property (weak) HIDConnection * connection;

@end

NS_ASSUME_NONNULL_END

#endif /* HIDMotionConnectionEventControl_h */
