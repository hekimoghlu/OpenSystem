/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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
#import <WebKitLegacy/WebHTMLView.h>
#if TARGET_OS_IPHONE
#import <WebKitLegacy/WAKView.h>
#endif

@class DOMDocumentFragment;
@class DOMNode;
@class DOMRange;

extern const float _WebHTMLViewPrintingMinimumShrinkFactor;
extern const float _WebHTMLViewPrintingMaximumShrinkFactor;

@interface WebHTMLView (WebPrivate)

+ (NSArray *)supportedMIMETypes;
+ (NSArray *)supportedMediaMIMETypes;
+ (NSArray *)supportedImageMIMETypes;
+ (NSArray *)supportedNonImageMIMETypes;
+ (NSArray *)unsupportedTextMIMETypes;

- (void)close;

#if !TARGET_OS_IPHONE

// Modifier (flagsChanged) tracking SPI
+ (void)_postFlagsChangedEvent:(NSEvent *)flagsChangedEvent;
- (void)_updateMouseoverWithFakeEvent;

- (void)_setAsideSubviews;
- (void)_restoreSubviews;

- (void)_updateMouseoverWithEvent:(NSEvent *)event;

+ (NSArray *)_insertablePasteboardTypes;
+ (NSString *)_dummyPasteboardType;
+ (NSArray *)_selectionPasteboardTypes;
- (void)_writeSelectionToPasteboard:(NSPasteboard *)pasteboard;

#endif

- (void)_frameOrBoundsChanged;

#if !TARGET_OS_IPHONE
- (void)_handleAutoscrollForMouseDragged:(NSEvent *)event;
#endif

// FIXME: _selectionRect is deprecated in favor of selectionRect, which is in protocol WebDocumentSelection.
// We can't remove this yet because it's still in use by Mail.
- (NSRect)_selectionRect;

- (BOOL)_canEdit;
- (BOOL)_canEditRichly;
- (BOOL)_canAlterCurrentSelection;
- (BOOL)_hasSelection;
- (BOOL)_hasSelectionOrInsertionPoint;
- (BOOL)_isEditable;

#if !TARGET_OS_IPHONE

- (BOOL)_transparentBackground;
- (void)_setTransparentBackground:(BOOL)isBackgroundTransparent;

#endif

- (void)_setToolTip:(NSString *)string;

#if !TARGET_OS_IPHONE

// SPI used by Mail.
// FIXME: These should all be moved to WebView; we won't always have a WebHTMLView.
- (NSImage *)_selectionDraggingImage;
- (NSRect)_selectionDraggingRect;
- (DOMNode *)_insertOrderedList;
- (DOMNode *)_insertUnorderedList;
- (BOOL)_canIncreaseSelectionListLevel;
- (BOOL)_canDecreaseSelectionListLevel;
- (DOMNode *)_increaseSelectionListLevel;
- (DOMNode *)_increaseSelectionListLevelOrdered;
- (DOMNode *)_increaseSelectionListLevelUnordered;
- (void)_decreaseSelectionListLevel;
- (DOMDocumentFragment *)_documentFragmentFromPasteboard:(NSPasteboard *)pasteboard forType:(NSString *)pboardType inContext:(DOMRange *)context subresources:(NSArray **)subresources;

#endif

- (BOOL)_isUsingAcceleratedCompositing;
#if TARGET_OS_IPHONE
- (WAKView *)_compositingLayersHostingView;
#else
- (NSView *)_compositingLayersHostingView;
#endif

#if !TARGET_OS_IPHONE
// SPI for printing (should be converted to API someday). When the WebHTMLView isn't being printed
// directly, this method must be called before paginating, or the computed height might be incorrect.
// Typically this would be called from inside an override of -[NSView knowsPageRange:].
- (void)_layoutForPrinting;
#endif
- (CGFloat)_adjustedBottomOfPageWithTop:(CGFloat)top bottom:(CGFloat)bottom limit:(CGFloat)bottomLimit;
- (BOOL)_isInPrintMode;
- (BOOL)_beginPrintModeWithPageWidth:(float)pageWidth height:(float)pageHeight shrinkToFit:(BOOL)shrinkToFit;
// Lays out to pages of the given minimum width and height or more (increasing both dimensions proportionally)
// as needed for the content to fit, but no more than the given maximum width.
- (BOOL)_beginPrintModeWithMinimumPageWidth:(CGFloat)minimumPageWidth height:(CGFloat)minimumPageHeight maximumPageWidth:(CGFloat)maximumPageWidth;
- (void)_endPrintMode;

- (BOOL)_isInScreenPaginationMode;
- (BOOL)_beginScreenPaginationModeWithPageSize:(CGSize)pageSize shrinkToFit:(BOOL)shrinkToFit;
- (void)_endScreenPaginationMode;

#if !TARGET_OS_IPHONE
- (BOOL)_canSmartReplaceWithPasteboard:(NSPasteboard *)pasteboard;
#endif

#if TARGET_OS_IPHONE
- (id)accessibilityRootElement;
#endif

#if !TARGET_OS_IPHONE
- (NSView *)_hitViewForEvent:(NSEvent *)event;
#endif

@end
