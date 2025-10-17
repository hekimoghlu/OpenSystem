/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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

//
//  IOHIDTelemetrySessionFilter.h
//  IOHIDTelemetrySessionFilter
//
//

#ifndef IOHIDTelemetrySessionFilter_h
#define IOHIDTelemetrySessionFilter_h

#import <HID/HID_Private.h>

NS_ASSUME_NONNULL_BEGIN

@interface IOHIDTelemetrySessionFilter : NSObject <HIDSessionFilter>

- (nullable instancetype)initWithSession:(HIDSession *)session;

- (nullable id)propertyForKey:(NSString *)key;

- (nullable HIDEvent *)filterEvent:(HIDEvent *)event
                        forService:(HIDEventService *)service;

- (void)activate;

- (void)serviceNotification:(HIDEventService *)service added:(BOOL)added;

- (BOOL)setProperty:(nullable id)value
             forKey:(NSString *)key;

- (void)setDispatchQueue:(dispatch_queue_t)queue;

- (nullable HIDEvent *)filterEvent:(HIDEvent *)event
                      toConnection:(HIDConnection *)connection
                       fromService:(HIDEventService *)service;


@end

NS_ASSUME_NONNULL_END

#endif /* IOHIDTelemetrySessionFilter_h */
