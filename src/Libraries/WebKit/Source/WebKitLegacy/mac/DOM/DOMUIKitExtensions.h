/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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
#if TARGET_OS_IPHONE

#import <WebKitLegacy/DOMElement.h>
#import <WebKitLegacy/DOMExtensions.h>
#import <WebKitLegacy/DOMHTMLAreaElement.h>
#import <WebKitLegacy/DOMHTMLImageElement.h>
#import <WebKitLegacy/DOMHTMLSelectElement.h>
#import <WebKitLegacy/DOMNode.h>
#import <WebKitLegacy/DOMRange.h>

typedef enum { 
    // The first four match SelectionDirection.  The last two don't have WebKit counterparts because
    // they aren't necessary until there is support vertical layout.
    WebTextAdjustmentForward,
    WebTextAdjustmentBackward,
    WebTextAdjustmentRight,
    WebTextAdjustmentLeft,
    WebTextAdjustmentUp,
    WebTextAdjustmentDown
} WebTextAdjustmentDirection; 

@interface DOMRange (UIKitExtensions)

- (void)move:(UInt32)amount inDirection:(WebTextAdjustmentDirection)direction;
- (void)extend:(UInt32)amount inDirection:(WebTextAdjustmentDirection)direction;
- (DOMNode *)firstNode;

@end

@interface DOMNode (UIKitExtensions)
- (NSArray *)borderRadii;
- (NSArray *)boundingBoxes;
- (NSArray *)absoluteQuads;     // return array of WKQuadObjects. takes transforms into account

- (BOOL)containsOnlyInlineObjects;
- (BOOL)isSelectableBlock;
- (DOMRange *)rangeOfContainingParagraph;
- (CGFloat)textHeight;
- (DOMNode *)findExplodedTextNodeAtPoint:(CGPoint)point;  // A second-chance pass to look for text nodes missed by the hit test.
@end

@interface DOMHTMLAreaElement (UIKitExtensions)
- (CGRect)boundingBoxWithOwner:(DOMNode *)anOwner;
- (WKQuad)absoluteQuadWithOwner:(DOMNode *)anOwner;     // takes transforms into account
- (NSArray *)boundingBoxesWithOwner:(DOMNode *)anOwner;
- (NSArray *)absoluteQuadsWithOwner:(DOMNode *)anOwner; // return array of WKQuadObjects. takes transforms into account
@end

@interface DOMHTMLSelectElement (UIKitExtensions)
- (unsigned)completeLength;
- (DOMNode *)listItemAtIndex:(int)anIndex;
@end

@interface DOMHTMLImageElement (WebDOMHTMLImageElementOperationsPrivate)
- (NSData *)dataRepresentation:(BOOL)rawImageData;
- (NSString *)mimeType;
@end

@interface DOMElement (DOMUIKitComplexityExtensions) 
- (int)structuralComplexityContribution; // Does not include children.
@end

#endif // TARGET_OS_IPHONE
