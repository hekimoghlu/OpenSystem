/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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
#import <WebKitLegacy/DOMNode.h>

#if TARGET_OS_IPHONE
#import <CoreGraphics/CoreGraphics.h>
#endif

@class DOMAttr;
@class DOMCSSStyleDeclaration;
@class DOMElement;
@class DOMNodeList;
@class NSString;

enum {
    DOM_ALLOW_KEYBOARD_INPUT = 1
} WEBKIT_ENUM_DEPRECATED_MAC(10_4, 10_14);

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMElement : DOMNode
@property (readonly, copy) NSString *tagName;
@property (readonly, strong) DOMCSSStyleDeclaration *style;
@property (readonly) int offsetLeft;
@property (readonly) int offsetTop;
@property (readonly) int offsetWidth;
@property (readonly) int offsetHeight;
@property (readonly) int clientLeft WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int clientTop WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) int clientWidth;
@property (readonly) int clientHeight;
@property int scrollLeft;
@property int scrollTop;
@property (readonly) int scrollWidth;
@property (readonly) int scrollHeight;
@property (readonly, strong) DOMElement *offsetParent;
@property (copy) NSString *innerHTML;
@property (copy) NSString *outerHTML;
@property (copy) NSString *className;
@property (readonly, copy) NSString *innerText WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, strong) DOMElement *previousElementSibling WEBKIT_AVAILABLE_MAC(10_6);
@property (readonly, strong) DOMElement *nextElementSibling WEBKIT_AVAILABLE_MAC(10_6);
@property (readonly, strong) DOMElement *firstElementChild WEBKIT_AVAILABLE_MAC(10_6);
@property (readonly, strong) DOMElement *lastElementChild WEBKIT_AVAILABLE_MAC(10_6);
@property (readonly) unsigned childElementCount WEBKIT_AVAILABLE_MAC(10_6);

#if TARGET_OS_IPHONE
@property (readonly) CGRect boundsInRootViewSpace;
#endif

- (NSString *)getAttribute:(NSString *)name;
- (void)setAttribute:(NSString *)name value:(NSString *)value WEBKIT_AVAILABLE_MAC(10_5);
- (void)removeAttribute:(NSString *)name;
- (DOMAttr *)getAttributeNode:(NSString *)name;
- (DOMAttr *)setAttributeNode:(DOMAttr *)newAttr;
- (DOMAttr *)removeAttributeNode:(DOMAttr *)oldAttr;
- (DOMNodeList *)getElementsByTagName:(NSString *)name;
- (NSString *)getAttributeNS:(NSString *)namespaceURI localName:(NSString *)localName WEBKIT_AVAILABLE_MAC(10_5);
- (void)setAttributeNS:(NSString *)namespaceURI qualifiedName:(NSString *)qualifiedName value:(NSString *)value WEBKIT_AVAILABLE_MAC(10_5);
- (void)removeAttributeNS:(NSString *)namespaceURI localName:(NSString *)localName WEBKIT_AVAILABLE_MAC(10_5);
- (DOMNodeList *)getElementsByTagNameNS:(NSString *)namespaceURI localName:(NSString *)localName WEBKIT_AVAILABLE_MAC(10_5);
- (DOMAttr *)getAttributeNodeNS:(NSString *)namespaceURI localName:(NSString *)localName WEBKIT_AVAILABLE_MAC(10_5);
- (DOMAttr *)setAttributeNodeNS:(DOMAttr *)newAttr;
- (BOOL)hasAttribute:(NSString *)name;
- (BOOL)hasAttributeNS:(NSString *)namespaceURI localName:(NSString *)localName WEBKIT_AVAILABLE_MAC(10_5);
- (void)focus WEBKIT_AVAILABLE_MAC(10_6);
- (void)blur WEBKIT_AVAILABLE_MAC(10_6);
- (void)scrollIntoView:(BOOL)alignWithTop WEBKIT_AVAILABLE_MAC(10_5);
- (void)scrollIntoViewIfNeeded:(BOOL)centerIfNeeded WEBKIT_AVAILABLE_MAC(10_5);
- (DOMNodeList *)getElementsByClassName:(NSString *)name WEBKIT_AVAILABLE_MAC(10_6);
#if !TARGET_OS_IPHONE
- (void)webkitRequestFullScreen:(unsigned short)flags WEBKIT_AVAILABLE_MAC(10_6);
#endif
- (DOMElement *)querySelector:(NSString *)selectors WEBKIT_AVAILABLE_MAC(10_6);
- (DOMNodeList *)querySelectorAll:(NSString *)selectors WEBKIT_AVAILABLE_MAC(10_6);
@end

@interface DOMElement (DOMElementDeprecated)
- (void)setAttribute:(NSString *)name :(NSString *)value WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (NSString *)getAttributeNS:(NSString *)namespaceURI :(NSString *)localName WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (void)setAttributeNS:(NSString *)namespaceURI :(NSString *)qualifiedName :(NSString *)value WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (void)removeAttributeNS:(NSString *)namespaceURI :(NSString *)localName WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (DOMNodeList *)getElementsByTagNameNS:(NSString *)namespaceURI :(NSString *)localName WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (DOMAttr *)getAttributeNodeNS:(NSString *)namespaceURI :(NSString *)localName WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (BOOL)hasAttributeNS:(NSString *)namespaceURI :(NSString *)localName WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (void)scrollByLines:(int)lines WEBKIT_DEPRECATED_MAC(10_5, 10_14);
- (void)scrollByPages:(int)pages WEBKIT_DEPRECATED_MAC(10_5, 10_14);
@end
