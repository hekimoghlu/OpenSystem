/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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
#import <Foundation/Foundation.h>

#import <WebKitLegacy/WebDocumentPrivate.h>

@class WebHTMLRepresentationPrivate;
#if !TARGET_OS_IPHONE
@class NSView;
#endif

@class DOMNode;
@class DOMElement;

@protocol WebDocumentMarkup;
@protocol WebDocumentRepresentation;
@protocol WebDocumentSourceRepresentation;

/*!
    @class WebHTMLRepresentation
*/
@interface WebHTMLRepresentation : NSObject <WebDocumentRepresentation, WebDocumentDOM>
{
@private
    WebHTMLRepresentationPrivate *_private;
}

+ (NSArray *)supportedMIMETypes;
+ (NSArray *)supportedMediaMIMETypes;
+ (NSArray *)supportedNonImageMIMETypes;
+ (NSArray *)supportedImageMIMETypes;
+ (NSArray *)unsupportedTextMIMETypes;

#if !TARGET_OS_IPHONE
- (NSAttributedString *)attributedStringFrom:(DOMNode *)startNode startOffset:(int)startOffset to:(DOMNode *)endNode endOffset:(int)endOffset;
#endif

- (DOMElement *)elementWithName:(NSString *)name inForm:(DOMElement *)form;
- (BOOL)elementDoesAutoComplete:(DOMElement *)element;
- (BOOL)elementIsPassword:(DOMElement *)element;
- (DOMElement *)formForElement:(DOMElement *)element;
- (DOMElement *)currentForm;
- (NSArray *)controlsInForm:(DOMElement *)form;
- (NSString *)searchForLabels:(NSArray *)labels beforeElement:(DOMElement *)element resultDistance:(NSUInteger*)outDistance resultIsInCellAbove:(BOOL*)outIsInCellAbove;
- (NSString *)matchLabels:(NSArray *)labels againstElement:(DOMElement *)element;

// Deprecated SPI
- (NSString *)searchForLabels:(NSArray *)labels beforeElement:(DOMElement *)element; // Use -searchForLabels:beforeElement:resultDistance:resultIsInCellAbove:

@end
