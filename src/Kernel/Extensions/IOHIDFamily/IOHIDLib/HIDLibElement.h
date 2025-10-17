/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
#ifndef IOHIDElement_h
#define IOHIDElement_h

#import <IOKit/hid/IOHIDKeys.h>
#import <IOKit/hid/IOHIDElement.h>
#import <IOKit/hid/IOHIDValue.h>
#import "IOHIDLibUserClient.h"

@interface HIDLibElement : NSObject {
    IOHIDElementRef     _element;
    IOHIDValueRef       _defaultValue;
    NSString            *_psKey;
    IOHIDElementStruct  _elementStruct;
    BOOL                _isConstant;
    BOOL                _isUpdated;
    
}

- (nullable instancetype)initWithElementRef:(nonnull IOHIDElementRef)elementRef;
- (nullable instancetype)initWithElementStruct:(nonnull IOHIDElementStruct *)elementStruct
                                        parent:(nullable IOHIDElementRef)parent
                                         index:(uint32_t)index;

- (void)setIntegerValue:(NSInteger)integerValue;

@property (nullable)        IOHIDElementRef     elementRef;
@property (nullable)        IOHIDValueRef       valueRef;
@property (nullable)        IOHIDValueRef       defaultValueRef;
@property                   NSInteger           integerValue;
@property (nullable)        NSData              *dataValue;
@property (nullable, copy)  NSString            *psKey;
@property (readonly)        uint64_t            timestamp;
@property (readonly)        NSInteger           length;
@property (readonly)        IOHIDElementStruct  elementStruct;
@property                   BOOL                isConstant;
@property                   BOOL                isUpdated;

/*
 * These properties can be predicated against using the kIOHIDElement keys.
 * The property names must be consistent with the key names in order for us
 * to be able to predicate properly.
 */
@property (readonly)        uint32_t                    elementCookie;
@property (readonly)        uint32_t                    collectionCookie;
@property (readonly)        IOHIDElementType            type;
@property (readonly)        uint32_t                    usage;
@property (readonly)        uint32_t                    usagePage;
@property (readonly)        uint32_t                    unit;
@property (readonly)        uint8_t                     unitExponent;
@property (readonly)        uint8_t                     reportID;
@property (readonly)        uint32_t                    usageMin;
@property (readonly)        uint32_t                    usageMax;
@property (readonly)        IOHIDElementCollectionType  collectionType;

@end

#endif /* IOHIDElement_h */
