/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 29, 2024.
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
#import <HID/HIDEventFields.h>
#import <HID/HIDEvent.h>

NS_ASSUME_NONNULL_BEGIN

/*!
 * @typedef HIDEventFieldInfoBlock
 *
 * @abstract
 * The block type used for enumerateFieldsWithBlock.
 */
typedef void (^HIDEventFieldInfoBlock) (HIDEventFieldInfo * eventField);

/*!
 * @category HIDEvent
 *
 * @abstract
 * HIDEvent extension to allow enumeration of
 */
@interface HIDEvent (HIDEventDesc)

/*!
 * @method enumerateFieldsWithBlock
 *
 * @abstract
 * Enumerates the event fields of the HIDEvent.
 *
 * @discussion
 * The block provided as a parameter is
 * called with a HIDEventFieldInfo argument describing each
 * field type.
 *
 * @param block
 * A block which will be called for each event field.
 */
- (void)enumerateFieldsWithBlock:(HIDEventFieldInfoBlock)block;

@end

NS_ASSUME_NONNULL_END
