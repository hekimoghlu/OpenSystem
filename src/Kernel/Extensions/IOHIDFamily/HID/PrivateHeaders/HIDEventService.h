/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 1, 2024.
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
#ifndef HIDEventService_h
#define HIDEventService_h

#import <Foundation/Foundation.h>
#import <HID/HIDBase.h>
#import <IOKit/hidobjc/HIDServiceBase.h>

NS_ASSUME_NONNULL_BEGIN

/*!
 * @category HIDEventService
 *
 * @abstract
 * Direct interaction with a HID service.
 *
 * @discussion
 * This should only be used by system code.
 */
@interface HIDEventService (HIDFramework)

- (instancetype)init NS_UNAVAILABLE;

/*!
 * @method propertyForKey
 *
 * @abstract
 * Obtains a property from the service.
 *
 * @param key
 * The property key.
 *
 * @result
 * Returns the property on success.
 */
- (nullable id)propertyForKey:(NSString *)key;

/*!
 * @method setProperty
 *
 * @abstract
 * Sets a property on the service.
 *
 * @param value
 * The value of the property.
 *
 * @param key
 * The property key.
 *
 * @result
 * Returns true on success.
 */
- (BOOL)setProperty:(nullable id)value forKey:(NSString *)key;

/*!
 * @method conformsToUsagePage
 *
 * @abstract
 * Iterates through the service's usage pairs to see if the service conforms to
 * the provided usage page and usage.
 *
 * @param usagePage
 * The device usage page.
 *
 * @param usage
 * The device usage.
 *
 * @result
 * Returns true if the service conforms to the provided usages.
 */
- (BOOL)conformsToUsagePage:(NSInteger)usagePage usage:(NSInteger)usage;

/*!
 * @method eventMatching
 *
 * @abstract
 * Queries the service for an event matching the criteria in the provided
 * dictionary.
 *
 * @param matching
 * Optional matching criteria that can be passed to the service.
 *
 * @result
 * Returns a HIDEvent on success.
 */
- (nullable HIDEvent *)eventMatching:(nullable NSDictionary *)matching;

/*!
 * @method registerWithSystem
 *
 * @abstract
 *  Triggers matching for a service with the current set of active clients
 *  Intended for use with "unregistered" services (through the kIOHIDServiceUnregisteredKey)
 */
- (void)registerWithSystem;

/*!
 * @property serviceID
 *
 * @abstract
 * The serviceID associated with the service.
 */
@property (readonly) uint64_t serviceID;

@end

NS_ASSUME_NONNULL_END

#endif /* HIDEventService_h */
