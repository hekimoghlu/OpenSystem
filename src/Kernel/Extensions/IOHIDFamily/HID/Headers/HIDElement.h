/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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
#ifndef HIDElement_h
#define HIDElement_h

#import <Foundation/Foundation.h>
#import <IOKit/hidobjc/HIDElementBase.h>

NS_ASSUME_NONNULL_BEGIN

/*!
 * @typedef HIDValueScaleType
 *
 * @abstract
 * Enumeration of HID value scale types.
 */
typedef NS_ENUM(NSInteger, HIDValueScaleType) {
    HIDValueScaleTypeCalibrated,
    HIDValueScaleTypePhysical,
    HIDValueScaleTypeExponent
};

/*!
 * @typedef HIDElementType
 *
 * @abstract
 * Enumeration of HID element types.
 */
typedef NS_ENUM(NSInteger, HIDElementType) {
    HIDElementTypeInputMisc   = 1,
    HIDElementTypeInputButton = 2,
    HIDElementTypeOutput      = 129,
    HIDElementTypeFeature     = 257,
    HIDElementTypeCollection  = 513
};

/*!
 * @category HIDElement
 *
 * @abstract
 * A representation of an individual element from a HID report descriptor
 * for a HID device.
 *
 * @discussion
 * A HIDElement is a way to interact with individual pieces of a HID report
 * descriptor. Every element has associated properties derived from the HID
 * descriptor and state that can be updated. HIDElements are associated with
 * and received from a HIDDevice. To update the state of a HIDElement, pull the
 * update using an input commit from the device and read the updated value using one
 * of the methods here, or set the value using one of the methods here and use an
 * output commit to update the device.
 *
 * A HIDElement should not be created, but received using other APIs.
 */
@interface HIDElement (HIDFramework)

- (instancetype)init NS_UNAVAILABLE;

/*!
 * @method scaleValue
 *
 * @abstract
 * Returns a scaled representation of the element's value based on the scale
 * type.
 *
 * @param type
 * The type of scaling to be performed.
 *
 * @result
 * A scaled floating point representation of the value.
 */
- (double)scaleValue:(HIDValueScaleType)type;

/*!
 * @property parent
 *
 * @abstract
 * The parent element (if any).
 */
@property (readonly, nullable) HIDElement * parent;

/*!
 * @property children
 *
 * @abstract
 * An array of child elements (if any).
 */
@property (readonly) NSArray<HIDElement *> * children;

/*!
 * @property integerValue
 *
 * @abstract
 * An integer representation of the element value.
 */
@property NSInteger integerValue;

/*!
 * @property dataValue
 *
 * @abstract
 * An NSData representation of the element value.
 */
@property (nullable, assign) NSData * dataValue;

/*!
 * @property type
 *
 * @abstract
 * The IOHIDElementType of the element.
 */
@property (readonly) HIDElementType type;

/*!
 * @property usagePage
 *
 * @abstract
 * The usage page of the element.
 */
@property (readonly) NSInteger usagePage;

/*!
 * @property usage
 *
 * @abstract
 * The usage of the element.
 */
@property (readonly) NSInteger usage;

/*!
 * @property reportID
 *
 * @abstract
 * The report ID associated with the element.
 */
@property (readonly) NSInteger reportID;

/*!
 * @property reportSize
 *
 * @abstract
 * The report size, in bits, of the element.
 */
@property (readonly) NSInteger reportSize;

/*!
 * @property unit
 *
 * @abstract
 * The unit of the element.
 */
@property (readonly) NSInteger unit;

/*!
 * @property unitExponent
 *
 * @abstract
 * The unit exponent of the element.
 */
@property (readonly) NSInteger unitExponent;

/*!
 * @property logicalMin
 *
 * @abstract
 * The logical minimum of the element.
 */
@property (readonly) NSInteger logicalMin;

/*!
 * @property logicalMax
 *
 * @abstract
 * The logical maximum of the element.
 */
@property (readonly) NSInteger logicalMax;

/*!
 * @property physicalMin
 *
 * @abstract
 * The physical minimum of the element.
 */
@property (readonly) NSInteger physicalMin;

/*!
 * @property physicalMax
 *
 * @abstract
 * The physical maximum of the element.
 */
@property (readonly) NSInteger physicalMax;

/*!
 * @property timestamp
 *
 * @abstract
 * The timestamp of the element in mach absolute time.
 */
@property (readonly) uint64_t timestamp;

@end

NS_ASSUME_NONNULL_END

#endif /* HIDElement_h */
