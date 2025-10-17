/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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

@class DOMCSSRule;
@class DOMCSSValue;
@class NSString;

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMCSSStyleDeclaration : DOMObject
@property (copy) NSString *cssText;
@property (readonly) unsigned length;
@property (readonly, strong) DOMCSSRule *parentRule;

- (NSString *)getPropertyValue:(NSString *)propertyName;
- (DOMCSSValue *)getPropertyCSSValue:(NSString *)propertyName;
- (NSString *)removeProperty:(NSString *)propertyName;
- (NSString *)getPropertyPriority:(NSString *)propertyName;
- (void)setProperty:(NSString *)propertyName value:(NSString *)value priority:(NSString *)priority WEBKIT_AVAILABLE_MAC(10_5);
- (NSString *)item:(unsigned)index;
- (NSString *)getPropertyShorthand:(NSString *)propertyName WEBKIT_DEPRECATED_MAC(10_5, 10_5);
- (BOOL)isPropertyImplicit:(NSString *)propertyName WEBKIT_AVAILABLE_MAC(10_5);
@end

@interface DOMCSSStyleDeclaration (DOMCSSStyleDeclarationDeprecated)
- (void)setProperty:(NSString *)propertyName :(NSString *)value :(NSString *)priority WEBKIT_DEPRECATED_MAC(10_4, 10_5);
@end
