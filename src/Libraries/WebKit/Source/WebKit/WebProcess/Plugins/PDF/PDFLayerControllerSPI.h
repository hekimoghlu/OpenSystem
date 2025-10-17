/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 19, 2025.
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
#pragma once

#if ENABLE(LEGACY_PDFKIT_PLUGIN)

#import <PDFKit/PDFKit.h>

@class PDFViewLayout;

typedef NS_ENUM(NSInteger, PDFLayerControllerCursorType) {
    kPDFLayerControllerCursorTypePointer = 0,
    kPDFLayerControllerCursorTypeHand,
    kPDFLayerControllerCursorTypeIBeam,
};

@protocol PDFLayerControllerDelegate <NSObject>

- (void)updateScrollPosition:(CGPoint)newPosition;
- (void)writeItemsToPasteboard:(NSArray *)items withTypes:(NSArray *)types;
- (void)showDefinitionForAttributedString:(NSAttributedString *)string atPoint:(CGPoint)point;
- (void)performWebSearch:(NSString *)string;
- (void)performSpotlightSearch:(NSString *)string;
- (void)openWithNativeApplication;
- (void)saveToPDF;

- (void)pdfLayerController:(PDFLayerController *)pdfLayerController didChangeActiveAnnotation:(PDFAnnotation *)annotation;
- (void)pdfLayerController:(PDFLayerController *)pdfLayerController clickedLinkWithURL:(NSURL *)url;
- (void)pdfLayerController:(PDFLayerController *)pdfLayerController didChangeContentScaleFactor:(CGFloat)scaleFactor;
- (void)pdfLayerController:(PDFLayerController *)pdfLayerController didChangeDisplayMode:(int)mode;
- (void)pdfLayerController:(PDFLayerController *)pdfLayerController didChangeSelection:(PDFSelection *)selection;

- (void)setMouseCursor:(PDFLayerControllerCursorType)cursorType;
- (void)didChangeAnnotationState;

@end

@interface PDFLayerController : NSObject
@end

@interface PDFLayerController ()

@property (retain) CALayer *parentLayer;
@property (retain) PDFDocument *document;
@property (retain) id<PDFLayerControllerDelegate> delegate;
@property (nonatomic, strong) NSString *URLFragment;
@property (nonatomic, class) bool useIOSurfaceForTiles;

- (void)setFrameSize:(CGSize)size;

- (PDFDisplayMode)displayMode;
- (void)setDisplayMode:(PDFDisplayMode)mode;
- (void)setDisplaysPageBreaks:(BOOL)pageBreaks;

- (CGFloat)contentScaleFactor;
- (void)setContentScaleFactor:(CGFloat)scaleFactor;

- (CGFloat)deviceScaleFactor;
- (void)setDeviceScaleFactor:(CGFloat)scaleFactor;

- (CGSize)contentSize;
- (CGSize)contentSizeRespectingZoom;

- (void)snapshotInContext:(CGContextRef)context;

- (void)setDisplaysPDFHUDController:(BOOL)displaysController;
- (void)zoomIn:(id)atPoint;
- (void)zoomOut:(id)atPoint;

- (void)magnifyWithMagnification:(CGFloat)magnification atPoint:(CGPoint)point immediately:(BOOL)immediately;

- (CGPoint)scrollPosition;
- (void)setScrollPosition:(CGPoint)newPosition;
- (void)scrollWithDelta:(CGSize)delta;

- (void)mouseDown:(NSEvent *)event;
- (void)rightMouseDown:(NSEvent *)event;
- (void)mouseMoved:(NSEvent *)event;
- (void)mouseUp:(NSEvent *)event;
- (void)mouseDragged:(NSEvent *)event;
- (void)mouseEntered:(NSEvent *)event;
- (void)mouseExited:(NSEvent *)event;

- (NSMenu *)menuForEvent:(NSEvent *)event withUserInterfaceLayoutDirection:(NSUserInterfaceLayoutDirection)direction;
- (NSMenu *)menuForEvent:(NSEvent *)event;

- (NSArray *)findString:(NSString *)string caseSensitive:(BOOL)isCaseSensitive highlightMatches:(BOOL)shouldHighlightMatches;

- (PDFSelection *)currentSelection;
- (void)setCurrentSelection:(PDFSelection *)selection;
- (PDFSelection *)searchSelection;
- (void)setSearchSelection:(PDFSelection *)selection;
- (void)gotoSelection:(PDFSelection *)selection;
- (PDFSelection *)getSelectionForWordAtPoint:(CGPoint)point;
- (NSArray *)rectsForSelectionInLayoutSpace:(PDFSelection *)selection;
- (NSArray *)rectsForAnnotationInLayoutSpace:(PDFAnnotation *)annotation;
- (PDFViewLayout *)layout;
- (NSRect)frame;

- (PDFPage *)currentPage;
- (NSUInteger)lastPageIndex;
- (NSUInteger)currentPageIndex;
- (void)gotoNextPage;
- (void)gotoPreviousPage;

- (void)copySelection;
- (void)selectAll;

- (bool)keyDown:(NSEvent *)event;

- (void)setHUDEnabled:(BOOL)enabled;
- (BOOL)hudEnabled;

- (CGRect)boundsForAnnotation:(PDFAnnotation *)annotation;
- (void)activateNextAnnotation:(BOOL)previous;

- (void)attemptToUnlockWithPassword:(NSString *)password;

- (void)searchInDictionaryWithSelection:(PDFSelection *)selection;

// Accessibility

- (id)accessibilityFocusedUIElement;
- (NSArray *)accessibilityAttributeNames;
- (BOOL)accessibilityIsAttributeSettable:(NSString *)attribute;
- (void)accessibilitySetValue:(id)value forAttribute:(NSString *)attribute;
- (NSArray *)accessibilityParameterizedAttributeNames;
- (NSString *)accessibilityRoleAttribute;
- (NSString *)accessibilityRoleDescriptionAttribute;
- (NSString *)accessibilityValueAttribute;
- (BOOL)accessibilityIsValueAttributeSettable;
- (NSString *)accessibilitySelectedTextAttribute;
- (BOOL)accessibilityIsSelectedTextAttributeSettable;
- (NSValue *)accessibilitySelectedTextRangeAttribute;
- (NSNumber *)accessibilityNumberOfCharactersAttribute;
- (BOOL)accessibilityIsNumberOfCharactersAttributeSettable;
- (NSValue *)accessibilityVisibleCharacterRangeAttribute;
- (BOOL)accessibilityIsVisibleCharacterRangeAttributeSettable;
- (NSNumber *)accessibilityLineForIndexAttributeForParameter:(id)parameter;
- (NSValue *)accessibilityRangeForLineAttributeForParameter:(id)parameter;
- (NSString *)accessibilityStringForRangeAttributeForParameter:(id)parameter;
- (NSValue *)accessibilityBoundsForRangeAttributeForParameter:(id)parameter;
- (NSArray *)accessibilityChildren;
- (void)setAccessibilityParent:(id)parent;
- (id)accessibilityElementForAnnotation:(PDFAnnotation *)annotation;
- (void)setDeviceColorSpace:(CGColorSpaceRef)colorSpace;

@end

#endif // ENABLE(LEGACY_PDFKIT_PLUGIN)
