/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
#ifndef HIDEventSystemClient_Internal_h
#define HIDEventSystemClient_Internal_h

#import <Foundation/Foundation.h>
#import <HID/HIDBase.h>
#import <HID/HIDEventSystemClient.h>
#import <IOKit/hidsystem/IOHIDEventSystemClient.h>

NS_ASSUME_NONNULL_BEGIN

@interface HIDEventSystemClient (priv)

@property (readonly) IOHIDEventSystemClientRef client;

/*!
 * @method initWithType:andAttributes
 *
 * @abstract
 * Creates a HIDEventSystemClient of the specified type.
 *
 * @discussion
 * A HIDEventSystem client is limited in its permitted functionality by the type
 * provided. A restriction due to lack of entitlement may not be immediately or
 * easily noticable, confer the HIDEventSystemClientType documentation above
 * for guidelines.
 *
 * @param type
 * The desired type of the client.
 *
 * @param attributes
 * Attributes to associate with the client.
 *
 * @result
 * A HIDEventSystemClient instance on success, nil on failure.
 */
- (nullable instancetype)initWithType:(HIDEventSystemClientType)type andAttributes:(NSDictionary * __nullable)attributes;

@end


NS_ASSUME_NONNULL_END

#endif /* HIDEventSystemClient_Internal_h */
