/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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
#import <WebKitLegacy/DOM.h>
#import <WebKitLegacy/WebAutocapitalizeTypes.h>
#import <WebKitLegacy/WebDOMOperationsPrivate.h>

#if TARGET_OS_IPHONE
#import <CoreText/CoreText.h>
#endif

@interface DOMNode (DOMNodeExtensionsPendingPublic)
#if !TARGET_OS_IPHONE
- (NSImage *)renderedImage;
#endif
- (NSArray *)textRects;
@end

@interface DOMNode (WebPrivate)
+ (id)_nodeFromJSWrapper:(JSObjectRef)jsWrapper;
- (void)getPreviewSnapshotImage:(CGImageRef*)cgImage andRects:(NSArray **)rects;
@end

// FIXME: this should be removed as soon as all internal Apple uses of it have been replaced with
// calls to the public method - (NSColor *)color.
@interface DOMRGBColor (WebPrivate)
#if !TARGET_OS_IPHONE
- (NSColor *)_color;
#endif
@end

// FIXME: this should be removed as soon as all internal Apple uses of it have been replaced with
// calls to the public method - (NSString *)text.
@interface DOMRange (WebPrivate)
- (NSString *)_text;
@end

@interface DOMRange (DOMRangeExtensions)
#if TARGET_OS_IPHONE
- (CGRect)boundingBox;
#else
- (NSRect)boundingBox;
#endif
#if !TARGET_OS_IPHONE
- (NSImage *)renderedImageForcingBlackText:(BOOL)forceBlackText;
#else
- (CGImageRef)renderedImageForcingBlackText:(BOOL)forceBlackText;
#endif
- (NSArray *)lineBoxRects; // Deprecated. Use textRects instead.
- (NSArray *)textRects;
@end

@interface DOMElement (WebPrivate)
#if !TARGET_OS_IPHONE
- (NSData *)_imageTIFFRepresentation;
#endif
- (CTFontRef)_font;
- (NSURL *)_getURLAttribute:(NSString *)name;
- (BOOL)isFocused;
@end

@interface DOMCSSStyleDeclaration (WebPrivate)
- (NSString *)_fontSizeDelta;
- (void)_setFontSizeDelta:(NSString *)fontSizeDelta;
@end

@interface DOMHTMLDocument (WebPrivate)
- (DOMDocumentFragment *)_createDocumentFragmentWithMarkupString:(NSString *)markupString baseURLString:(NSString *)baseURLString;
- (DOMDocumentFragment *)_createDocumentFragmentWithText:(NSString *)text;
@end

@interface DOMHTMLTableCellElement (WebPrivate)
- (DOMHTMLTableCellElement *)_cellAbove;
@end

@interface DOMHTMLInputElement (FormAutoFillTransition)
- (BOOL)_isTextField;
@end

#if TARGET_OS_IPHONE
// These changes are necessary to detect whether a form input was modified by a user
// or javascript
@interface DOMHTMLInputElement (FormPromptAdditions)
- (BOOL)_isEdited;
@end

@interface DOMHTMLTextAreaElement (FormPromptAdditions)
- (BOOL)_isEdited;
@end
#endif // TARGET_OS_IPHONE

@interface DOMHTMLSelectElement (FormAutoFillTransition)
- (void)_activateItemAtIndex:(int)index;
- (void)_activateItemAtIndex:(int)index allowMultipleSelection:(BOOL)allowMultipleSelection;
@end

#if TARGET_OS_IPHONE
enum { WebMediaQueryOrientationCurrent, WebMediaQueryOrientationPortrait, WebMediaQueryOrientationLandscape };
@interface DOMHTMLLinkElement (WebPrivate)
- (BOOL)_mediaQueryMatchesForOrientation:(int)orientation;
- (BOOL)_mediaQueryMatches;
@end

// These changes are useful to get the AutocapitalizeType on particular form controls.
@interface DOMHTMLInputElement (AutocapitalizeAdditions)
- (WebAutocapitalizeType)_autocapitalizeType;
@end

@interface DOMHTMLTextAreaElement (AutocapitalizeAdditions)
- (WebAutocapitalizeType)_autocapitalizeType;
@end

// These are used by Date and Time input types because the generated ObjC methods default to not dispatching events.
@interface DOMHTMLInputElement (WebInputChangeEventAdditions)
- (void)setValueWithChangeEvent:(NSString *)newValue;
- (void)setValueAsNumberWithChangeEvent:(double)newValueAsNumber;
@end
#endif // TARGET_OS_IPHONE
