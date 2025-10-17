/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 29, 2022.
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

@class DOMNode;
@class NSString;

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMNamedNodeMap : DOMObject
@property (readonly) unsigned length;

- (DOMNode *)getNamedItem:(NSString *)name;
- (DOMNode *)setNamedItem:(DOMNode *)node;
- (DOMNode *)removeNamedItem:(NSString *)name;
- (DOMNode *)item:(unsigned)index;
- (DOMNode *)getNamedItemNS:(NSString *)namespaceURI localName:(NSString *)localName WEBKIT_AVAILABLE_MAC(10_5);
- (DOMNode *)setNamedItemNS:(DOMNode *)node;
- (DOMNode *)removeNamedItemNS:(NSString *)namespaceURI localName:(NSString *)localName WEBKIT_AVAILABLE_MAC(10_5);
@end

@interface DOMNamedNodeMap (DOMNamedNodeMapDeprecated)
- (DOMNode *)getNamedItemNS:(NSString *)namespaceURI :(NSString *)localName WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (DOMNode *)removeNamedItemNS:(NSString *)namespaceURI :(NSString *)localName WEBKIT_DEPRECATED_MAC(10_4, 10_5);
@end
