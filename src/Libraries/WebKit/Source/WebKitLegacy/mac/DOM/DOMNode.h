/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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
#import <WebKitLegacy/DOMEventTarget.h>

@class DOMDocument;
@class DOMElement;
@class DOMNamedNodeMap;
@class DOMNode;
@class DOMNodeList;
@class NSString;

enum {
    DOM_ELEMENT_NODE = 1,
    DOM_ATTRIBUTE_NODE = 2,
    DOM_TEXT_NODE = 3,
    DOM_CDATA_SECTION_NODE = 4,
    DOM_ENTITY_REFERENCE_NODE = 5,
    DOM_ENTITY_NODE = 6,
    DOM_PROCESSING_INSTRUCTION_NODE = 7,
    DOM_COMMENT_NODE = 8,
    DOM_DOCUMENT_NODE = 9,
    DOM_DOCUMENT_TYPE_NODE = 10,
    DOM_DOCUMENT_FRAGMENT_NODE = 11,
    DOM_NOTATION_NODE = 12,
    DOM_DOCUMENT_POSITION_DISCONNECTED = 0x01,
    DOM_DOCUMENT_POSITION_PRECEDING = 0x02,
    DOM_DOCUMENT_POSITION_FOLLOWING = 0x04,
    DOM_DOCUMENT_POSITION_CONTAINS = 0x08,
    DOM_DOCUMENT_POSITION_CONTAINED_BY = 0x10,
    DOM_DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC = 0x20
} WEBKIT_ENUM_DEPRECATED_MAC(10_4, 10_14);

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMNode : DOMObject <DOMEventTarget>
@property (readonly, copy) NSString *nodeName;
@property (copy) NSString *nodeValue;
@property (readonly) unsigned short nodeType;
@property (readonly, strong) DOMNode *parentNode;
@property (readonly, strong) DOMNodeList *childNodes;
@property (readonly, strong) DOMNode *firstChild;
@property (readonly, strong) DOMNode *lastChild;
@property (readonly, strong) DOMNode *previousSibling;
@property (readonly, strong) DOMNode *nextSibling;
@property (readonly, strong) DOMDocument *ownerDocument;
@property (readonly, copy) NSString *namespaceURI;
@property (copy) NSString *prefix;
@property (readonly, copy) NSString *localName;
@property (readonly, strong) DOMNamedNodeMap *attributes;
@property (readonly, copy) NSString *baseURI WEBKIT_AVAILABLE_MAC(10_5);
@property (copy) NSString *textContent WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, strong) DOMElement *parentElement WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly) BOOL isContentEditable WEBKIT_AVAILABLE_MAC(10_5);

- (DOMNode *)insertBefore:(DOMNode *)newChild refChild:(DOMNode *)refChild WEBKIT_AVAILABLE_MAC(10_5);
- (DOMNode *)replaceChild:(DOMNode *)newChild oldChild:(DOMNode *)oldChild WEBKIT_AVAILABLE_MAC(10_5);
- (DOMNode *)removeChild:(DOMNode *)oldChild;
- (DOMNode *)appendChild:(DOMNode *)newChild;
- (BOOL)hasChildNodes;
- (DOMNode *)cloneNode:(BOOL)deep;
- (void)normalize;
- (BOOL)isSupported:(NSString *)feature version:(NSString *)version WEBKIT_AVAILABLE_MAC(10_5);
- (BOOL)hasAttributes;
- (BOOL)isSameNode:(DOMNode *)other WEBKIT_AVAILABLE_MAC(10_5);
- (BOOL)isEqualNode:(DOMNode *)other WEBKIT_AVAILABLE_MAC(10_5);
- (NSString *)lookupPrefix:(NSString *)namespaceURI WEBKIT_AVAILABLE_MAC(10_5);
- (NSString *)lookupNamespaceURI:(NSString *)prefix WEBKIT_AVAILABLE_MAC(10_5);
- (BOOL)isDefaultNamespace:(NSString *)namespaceURI WEBKIT_AVAILABLE_MAC(10_5);
- (unsigned short)compareDocumentPosition:(DOMNode *)other WEBKIT_AVAILABLE_MAC(10_6);
- (BOOL)contains:(DOMNode *)other WEBKIT_AVAILABLE_MAC(10_5);
@end

@interface DOMNode (DOMNodeDeprecated)
- (DOMNode *)insertBefore:(DOMNode *)newChild :(DOMNode *)refChild WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (DOMNode *)replaceChild:(DOMNode *)newChild :(DOMNode *)oldChild WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (BOOL)isSupported:(NSString *)feature :(NSString *)version WEBKIT_DEPRECATED_MAC(10_4, 10_5);
@end
