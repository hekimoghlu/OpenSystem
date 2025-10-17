/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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
#import <WebKitLegacy/DOMCSSValue.h>

@class DOMCounter;
@class DOMRGBColor;
@class DOMRect;
@class NSString;

enum {
    DOM_CSS_UNKNOWN = 0,
    DOM_CSS_NUMBER = 1,
    DOM_CSS_PERCENTAGE = 2,
    DOM_CSS_EMS = 3,
    DOM_CSS_EXS = 4,
    DOM_CSS_PX = 5,
    DOM_CSS_CM = 6,
    DOM_CSS_MM = 7,
    DOM_CSS_IN = 8,
    DOM_CSS_PT = 9,
    DOM_CSS_PC = 10,
    DOM_CSS_DEG = 11,
    DOM_CSS_RAD = 12,
    DOM_CSS_GRAD = 13,
    DOM_CSS_MS = 14,
    DOM_CSS_S = 15,
    DOM_CSS_HZ = 16,
    DOM_CSS_KHZ = 17,
    DOM_CSS_DIMENSION = 18,
    DOM_CSS_STRING = 19,
    DOM_CSS_URI = 20,
    DOM_CSS_IDENT = 21,
    DOM_CSS_ATTR = 22,
    DOM_CSS_COUNTER = 23,
    DOM_CSS_RECT = 24,
    DOM_CSS_RGBCOLOR = 25,
    DOM_CSS_VW = 26,
    DOM_CSS_VH = 27,
    DOM_CSS_VMIN = 28,
    DOM_CSS_VMAX = 29
} WEBKIT_ENUM_DEPRECATED_MAC(10_4, 10_14);

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMCSSPrimitiveValue : DOMCSSValue
@property (readonly) unsigned short primitiveType;

- (void)setFloatValue:(unsigned short)unitType floatValue:(float)floatValue WEBKIT_AVAILABLE_MAC(10_5);
- (float)getFloatValue:(unsigned short)unitType;
- (void)setStringValue:(unsigned short)stringType stringValue:(NSString *)stringValue WEBKIT_AVAILABLE_MAC(10_5);
- (NSString *)getStringValue;
- (DOMCounter *)getCounterValue;
- (DOMRect *)getRectValue;
- (DOMRGBColor *)getRGBColorValue;
@end

@interface DOMCSSPrimitiveValue (DOMCSSPrimitiveValueDeprecated)
- (void)setFloatValue:(unsigned short)unitType :(float)floatValue WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (void)setStringValue:(unsigned short)stringType :(NSString *)stringValue WEBKIT_DEPRECATED_MAC(10_4, 10_5);
@end
