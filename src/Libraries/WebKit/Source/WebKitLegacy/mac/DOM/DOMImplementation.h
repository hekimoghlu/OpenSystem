/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 28, 2023.
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

@class DOMCSSStyleSheet;
@class DOMDocument;
@class DOMDocumentType;
@class DOMHTMLDocument;
@class NSString;

WEBKIT_CLASS_DEPRECATED_MAC(10_4, 10_14)
@interface DOMImplementation : DOMObject
- (BOOL)hasFeature:(NSString *)feature version:(NSString *)version WEBKIT_AVAILABLE_MAC(10_5);
- (DOMDocumentType *)createDocumentType:(NSString *)qualifiedName publicId:(NSString *)publicId systemId:(NSString *)systemId WEBKIT_AVAILABLE_MAC(10_5);
- (DOMDocument *)createDocument:(NSString *)namespaceURI qualifiedName:(NSString *)qualifiedName doctype:(DOMDocumentType *)doctype WEBKIT_AVAILABLE_MAC(10_5);
- (DOMCSSStyleSheet *)createCSSStyleSheet:(NSString *)title media:(NSString *)media WEBKIT_AVAILABLE_MAC(10_5);
- (DOMHTMLDocument *)createHTMLDocument:(NSString *)title WEBKIT_AVAILABLE_MAC(10_5);
@end

@interface DOMImplementation (DOMImplementationDeprecated)
- (BOOL)hasFeature:(NSString *)feature :(NSString *)version WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (DOMDocumentType *)createDocumentType:(NSString *)qualifiedName :(NSString *)publicId :(NSString *)systemId WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (DOMDocument *)createDocument:(NSString *)namespaceURI :(NSString *)qualifiedName :(DOMDocumentType *)doctype WEBKIT_DEPRECATED_MAC(10_4, 10_5);
- (DOMCSSStyleSheet *)createCSSStyleSheet:(NSString *)title :(NSString *)media WEBKIT_DEPRECATED_MAC(10_4, 10_5);
@end
