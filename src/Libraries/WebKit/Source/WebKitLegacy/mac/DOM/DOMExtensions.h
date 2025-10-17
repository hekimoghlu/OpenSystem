/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
#import <CoreGraphics/CoreGraphics.h>
#import <WebKitLegacy/DOMCSS.h>
#import <WebKitLegacy/DOMHTML.h>
#import <WebKitLegacy/DOMRange.h>

@class NSArray;
@class NSImage;
@class NSURL;

#if TARGET_OS_IPHONE

@interface DOMHTMLElement (DOMHTMLElementExtensions)
- (int)scrollXOffset;
- (int)scrollYOffset;
- (void)setScrollXOffset:(int)x scrollYOffset:(int)y;
- (void)setScrollXOffset:(int)x scrollYOffset:(int)y adjustForIOSCaret:(BOOL)adjustForIOSCaret;
- (void)absolutePosition:(int*)x :(int*)y :(int*)w :(int*)h;
@end

typedef struct _WKQuad {
    CGPoint p1;
    CGPoint p2;
    CGPoint p3;
    CGPoint p4;
} WKQuad;

@interface WKQuadObject : NSObject
- (id)initWithQuad:(WKQuad)quad;
- (WKQuad)quad;
- (CGRect)boundingBox;
@end

#endif

@interface DOMNode (DOMNodeExtensions)
#if TARGET_OS_IPHONE
- (CGRect)boundingBox;
#else
- (NSRect)boundingBox WEBKIT_AVAILABLE_MAC(10_5);
#endif
- (NSArray *)lineBoxRects WEBKIT_AVAILABLE_MAC(10_5);

#if TARGET_OS_IPHONE
- (CGRect)boundingBoxUsingTransforms; // takes transforms into account

- (WKQuad)absoluteQuad;
- (WKQuad)absoluteQuadAndInsideFixedPosition:(BOOL *)insideFixed;
- (NSArray *)lineBoxQuads; // returns array of WKQuadObject

- (NSURL *)hrefURL;
- (CGRect)hrefFrame;
- (NSString *)hrefTarget;
- (NSString *)hrefLabel;
- (NSString *)hrefTitle;
- (CGRect)boundingFrame;
- (WKQuad)innerFrameQuad; // takes transforms into account
- (float)computedFontSize;
- (DOMNode *)nextFocusNode;
- (DOMNode *)previousFocusNode;
#endif
@end

@interface DOMElement (DOMElementAppKitExtensions)
#if !TARGET_OS_IPHONE
- (NSImage *)image WEBKIT_AVAILABLE_MAC(10_5);
#endif
@end

@interface DOMHTMLDocument (DOMHTMLDocumentExtensions)
- (DOMDocumentFragment *)createDocumentFragmentWithMarkupString:(NSString *)markupString baseURL:(NSURL *)baseURL WEBKIT_AVAILABLE_MAC(10_5);
- (DOMDocumentFragment *)createDocumentFragmentWithText:(NSString *)text WEBKIT_AVAILABLE_MAC(10_5);
@end

#if TARGET_OS_IPHONE

@interface DOMHTMLAreaElement (DOMHTMLAreaElementExtensions)
- (CGRect)boundingFrameForOwner:(DOMNode *)anOwner;
@end

@interface DOMHTMLSelectElement (DOMHTMLSelectElementExtensions)
- (DOMNode *)listItemAtIndex:(int)anIndex;
@end

#endif
