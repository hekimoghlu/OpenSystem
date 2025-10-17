/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 16, 2023.
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
#ifndef HIDSession_h
#define HIDSession_h

#import <Foundation/Foundation.h>
#import <HID/HIDBase.h>
#import <IOKit/hidobjc/HIDSessionBase.h>

NS_ASSUME_NONNULL_BEGIN

/*!
 * @category HIDSession
 *
 * @abstract
 * Direct interaction with the HID session.
 *
 * @discussion
 * Should only be used by system code.
 */
@interface HIDSession (HIDFramework)

- (instancetype)init NS_UNAVAILABLE;

/*!
 * @method propertyForKey
 *
 * @abstract
 * Obtains a property from the session.
 *
 * @param key
 * The property key to query.
 *
 * @result
 * The property on success, nil on failure.
 */
- (nullable id)propertyForKey:(NSString *)key;

/*!
 * @method setProperty
 *
 * @abstract
 * Sets a property on the session.
 *
 * @param value
 * The value of the property to set.
 *
 * @param key
 * The property key to set.
 *
 * @result
 * true on success, nil on failure.
 */
- (BOOL)setProperty:(nullable id)value forKey:(NSString *)key;

@end

NS_ASSUME_NONNULL_END

#endif /* HIDSession_h */
