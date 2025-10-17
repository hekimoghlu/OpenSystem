/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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
#import <WebKitLegacy/DOMObject.h>

@class DOMCSSPrimitiveValue;
#if !TARGET_OS_IPHONE
@class NSColor;
#else
typedef struct CGColor* CGColorRef;
#endif

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMRGBColor : DOMObject
@property (readonly, strong) DOMCSSPrimitiveValue *red;
@property (readonly, strong) DOMCSSPrimitiveValue *green;
@property (readonly, strong) DOMCSSPrimitiveValue *blue;
@property (readonly, strong) DOMCSSPrimitiveValue *alpha;
#if !TARGET_OS_IPHONE
@property (readonly, copy) NSColor *color WEBKIT_AVAILABLE_MAC(10_5);
#else
- (CGColorRef)color;
#endif
@end
