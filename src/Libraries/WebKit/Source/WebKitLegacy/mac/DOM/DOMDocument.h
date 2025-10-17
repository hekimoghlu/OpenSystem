/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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

@class DOMAbstractView;
@class DOMAttr;
@class DOMCDATASection;
@class DOMCSSRuleList;
@class DOMCSSStyleDeclaration;
@class DOMComment;
@class DOMDocumentFragment;
@class DOMDocumentType;
@class DOMElement;
@class DOMEntityReference;
@class DOMEvent;
@class DOMHTMLCollection;
@class DOMHTMLElement;
@class DOMImplementation;
@class DOMNode;
@class DOMNodeIterator;
@class DOMNodeList;
@class DOMProcessingInstruction;
@class DOMRange;
@class DOMStyleSheetList;
@class DOMText;
@class DOMTreeWalker;
@class DOMXPathExpression;
@class DOMXPathResult;
@class NSString;
@protocol DOMNodeFilter;
@protocol DOMXPathNSResolver;

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMDocument : DOMNode
@property (readonly, strong) DOMDocumentType *doctype;
@property (readonly, strong) DOMImplementation *implementation;
@property (readonly, strong) DOMElement *documentElement;
@property (readonly, copy) NSString *inputEncoding WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, copy) NSString *xmlEncoding WEBKIT_AVAILABLE_MAC(10_5);
@property (copy) NSString *xmlVersion WEBKIT_AVAILABLE_MAC(10_5);
@property BOOL xmlStandalone WEBKIT_AVAILABLE_MAC(10_5);
@property (copy) NSString *documentURI WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, strong) DOMAbstractView *defaultView;
@property (readonly, strong) DOMStyleSheetList *styleSheets;
@property (copy) NSString *title;
@property (readonly, copy) NSString *referrer;
@property (readonly, copy) NSString *domain;
@property (readonly, copy) NSString *URL;
@property (copy) NSString *cookie;
@property (strong) DOMHTMLElement *body;
@property (readonly, strong) DOMHTMLCollection *images;
@property (readonly, strong) DOMHTMLCollection *applets;
@property (readonly, strong) DOMHTMLCollection *links;
@property (readonly, strong) DOMHTMLCollection *forms;
@property (readonly, strong) DOMHTMLCollection *anchors;
@property (readonly, copy) NSString *lastModified WEBKIT_AVAILABLE_MAC(10_6);
@property (copy) NSString *charset WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, copy) NSString *defaultCharset WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, copy) NSString *readyState WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, copy) NSString *characterSet WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, copy) NSString *preferredStylesheetSet WEBKIT_AVAILABLE_MAC(10_5);
@property (copy) NSString *selectedStylesheetSet WEBKIT_AVAILABLE_MAC(10_5);
@property (readonly, strong) DOMElement *activeElement WEBKIT_AVAILABLE_MAC(10_6);

- (DOMElement *)createElement:(NSString *)tagName;
- (DOMDocumentFragment *)createDocumentFragment;
- (DOMText *)createTextNode:(NSString *)data;
- (DOMComment *)createComment:(NSString *)data;
- (DOMCDATASection *)createCDATASection:(NSString *)data;
- (DOMProcessingInstruction *)createProcessingInstruction:(NSString *)target data:(NSString *)data WEBKIT_AVAILABLE_MAC(10_5);
- (DOMAttr *)createAttribute:(NSString *)name;
- (DOMEntityReference *)createEntityReference:(NSString *)name;
- (DOMNodeList *)getElementsByTagName:(NSString *)tagname;
- (DOMNode *)importNode:(DOMNode *)importedNode deep:(BOOL)deep WEBKIT_AVAILABLE_MAC(10_5);
- (DOMElement *)createElementNS:(NSString *)namespaceURI qualifiedName:(NSString *)qualifiedName WEBKIT_AVAILABLE_MAC(10_5);
- (DOMAttr *)createAttributeNS:(NSString *)namespaceURI qualifiedName:(NSString *)qualifiedName WEBKIT_AVAILABLE_MAC(10_5);
- (DOMNodeList *)getElementsByTagNameNS:(NSString *)namespaceURI localName:(NSString *)localName WEBKIT_AVAILABLE_MAC(10_5);
- (DOMNode *)adoptNode:(DOMNode *)source WEBKIT_AVAILABLE_MAC(10_5);
- (DOMEvent *)createEvent:(NSString *)eventType;
- (DOMRange *)createRange;
- (DOMNodeIterator *)createNodeIterator:(DOMNode *)root whatToShow:(unsigned)whatToShow filter:(id <DOMNodeFilter>)filter expandEntityReferences:(BOOL)expandEntityReferences WEBKIT_AVAILABLE_MAC(10_5);
- (DOMTreeWalker *)createTreeWalker:(DOMNode *)root whatToShow:(unsigned)whatToShow filter:(id <DOMNodeFilter>)filter expandEntityReferences:(BOOL)expandEntityReferences WEBKIT_AVAILABLE_MAC(10_5);
- (DOMCSSStyleDeclaration *)getOverrideStyle:(DOMElement *)element pseudoElement:(NSString *)pseudoElement WEBKIT_AVAILABLE_MAC(10_5);
- (DOMXPathExpression *)createExpression:(NSString *)expression resolver:(id <DOMXPathNSResolver>)resolver WEBKIT_AVAILABLE_MAC(10_5);
- (id <DOMXPathNSResolver>)createNSResolver:(DOMNode *)nodeResolver WEBKIT_AVAILABLE_MAC(10_5);
- (DOMXPathResult *)evaluate:(NSString *)expression contextNode:(DOMNode *)contextNode resolver:(id <DOMXPathNSResolver>)resolver type:(unsigned short)type inResult:(DOMXPathResult *)inResult WEBKIT_AVAILABLE_MAC(10_5);
- (BOOL)execCommand:(NSString *)command userInterface:(BOOL)userInterface value:(NSString *)value WEBKIT_AVAILABLE_MAC(10_5);
- (BOOL)execCommand:(NSString *)command userInterface:(BOOL)userInterface WEBKIT_AVAILABLE_MAC(10_5);
- (BOOL)execCommand:(NSString *)command WEBKIT_AVAILABLE_MAC(10_5);
- (BOOL)queryCommandEnabled:(NSString *)command WEBKIT_AVAILABLE_MAC(10_5);
- (BOOL)queryCommandIndeterm:(NSString *)command WEBKIT_AVAILABLE_MAC(10_5);
- (BOOL)queryCommandState:(NSString *)command WEBKIT_AVAILABLE_MAC(10_5);
- (BOOL)queryCommandSupported:(NSString *)command WEBKIT_AVAILABLE_MAC(10_5);
- (NSString *)queryCommandValue:(NSString *)command WEBKIT_AVAILABLE_MAC(10_5);
- (DOMNodeList *)getElementsByName:(NSString *)elementName;
- (DOMElement *)elementFromPoint:(int)x y:(int)y WEBKIT_AVAILABLE_MAC(10_5);
- (DOMCSSStyleDeclaration *)createCSSStyleDeclaration WEBKIT_AVAILABLE_MAC(10_5);
- (DOMCSSStyleDeclaration *)getComputedStyle:(DOMElement *)element pseudoElement:(NSString *)pseudoElement WEBKIT_AVAILABLE_MAC(10_5);
- (DOMCSSRuleList *)getMatchedCSSRules:(DOMElement *)element pseudoElement:(NSString *)pseudoElement WEBKIT_AVAILABLE_MAC(10_5);
- (DOMCSSRuleList *)getMatchedCSSRules:(DOMElement *)element pseudoElement:(NSString *)pseudoElement authorOnly:(BOOL)authorOnly WEBKIT_AVAILABLE_MAC(10_5);
- (DOMNodeList *)getElementsByClassName:(NSString *)classNames WEBKIT_AVAILABLE_MAC(10_6);
- (BOOL)hasFocus WEBKIT_AVAILABLE_MAC(10_6);
#if !TARGET_OS_IPHONE
- (void)webkitCancelFullScreen WEBKIT_AVAILABLE_MAC(10_6);
#endif
- (DOMElement *)getElementById:(NSString *)elementId;
- (DOMElement *)querySelector:(NSString *)selectors WEBKIT_AVAILABLE_MAC(10_6);
- (DOMNodeList *)querySelectorAll:(NSString *)selectors WEBKIT_AVAILABLE_MAC(10_6);
@end

@interface DOMDocument (DOMDocumentDeprecated)
- (DOMProcessingInstruction *)createProcessingInstruction:(NSString *)target :(NSString *)data WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (DOMNode *)importNode:(DOMNode *)importedNode :(BOOL)deep WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (DOMElement *)createElementNS:(NSString *)namespaceURI :(NSString *)qualifiedName WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (DOMAttr *)createAttributeNS:(NSString *)namespaceURI :(NSString *)qualifiedName WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (DOMNodeList *)getElementsByTagNameNS:(NSString *)namespaceURI :(NSString *)localName WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (DOMNodeIterator *)createNodeIterator:(DOMNode *)root :(unsigned)whatToShow :(id <DOMNodeFilter>)filter :(BOOL)expandEntityReferences WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (DOMTreeWalker *)createTreeWalker:(DOMNode *)root :(unsigned)whatToShow :(id <DOMNodeFilter>)filter :(BOOL)expandEntityReferences WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (DOMCSSStyleDeclaration *)getOverrideStyle:(DOMElement *)element :(NSString *)pseudoElement WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (DOMXPathExpression *)createExpression:(NSString *)expression :(id <DOMXPathNSResolver>)resolver WEBKIT_DEPRECATED_MAC(10_5, 10_5);
- (DOMXPathResult *)evaluate:(NSString *)expression :(DOMNode *)contextNode :(id <DOMXPathNSResolver>)resolver :(unsigned short)type :(DOMXPathResult *)inResult WEBKIT_DEPRECATED_MAC(10_5, 10_5);
- (DOMCSSStyleDeclaration *)getComputedStyle:(DOMElement *)element :(NSString *)pseudoElement WEBKIT_DEPRECATED_MAC(10_4, 10_5);
@end
