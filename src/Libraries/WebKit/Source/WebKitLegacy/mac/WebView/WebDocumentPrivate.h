/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 7, 2022.
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
#import <WebKitLegacy/WebDocument.h>
#import <WebKitLegacy/WebFrame.h>
#import <WebKitLegacy/WebHTMLView.h>
#if TARGET_OS_IPHONE
#import <WebKitLegacy/WAKView.h>
#endif

@class DOMDocument;
@class PDFDocument;

#if !TARGET_OS_IPHONE
@protocol WebDocumentImage <NSObject>
- (NSImage *)image;
@end
#endif

// This method is deprecated as it now lives on WebFrame.
@protocol WebDocumentDOM <NSObject>
- (DOMDocument *)DOMDocument;
- (BOOL)canSaveAsWebArchive;
@end

@protocol WebDocumentSelection <WebDocumentText>
#if !TARGET_OS_IPHONE
- (NSArray *)pasteboardTypesForSelection;
- (void)writeSelectionWithPasteboardTypes:(NSArray *)types toPasteboard:(NSPasteboard *)pasteboard;
#endif

// Array of rects that tightly enclose the selected text, in coordinates of selectinView.
- (NSArray *)selectionTextRects;

// Rect tightly enclosing the entire selected area, in coordinates of selectionView.
- (NSRect)selectionRect;

#if !TARGET_OS_IPHONE
// NSImage of the portion of the selection that's in view. This does not draw backgrounds. 
// The text is all black according to the parameter.
- (NSImage *)selectionImageForcingBlackText:(BOOL)forceBlackText;
#else
- (CGImageRef)selectionImageForcingBlackText:(BOOL)forceBlackText;
#endif

// Rect tightly enclosing the entire selected area, in coordinates of selectionView.
// NOTE: This method is equivalent to selectionRect and shouldn't be used; use selectionRect instead.
- (NSRect)selectionImageRect;

// View that draws the selection and can be made first responder. Often this is self but it could be
// a nested view, as for example in the case of WebPDFView.
#if TARGET_OS_IPHONE
- (WAKView *)selectionView;
#else
- (NSView *)selectionView;
#endif
@end

@protocol WebDocumentPDF <WebDocumentText>
- (PDFDocument *)PDFDocument;
@end

@protocol WebDocumentIncrementalSearching
/*!
@method searchFor:direction:caseSensitive:wrap:startInSelection:
 @abstract Searches a document view for a string and highlights the string if it is found.
 @param string The string to search for.
 @param forward YES to search forward, NO to seach backwards.
 @param caseFlag YES to for case-sensitive search, NO for case-insensitive search.
 @param wrapFlag YES to wrap around, NO to avoid wrapping.
 @param startInSelection YES to begin search in the selected text (useful for incremental searching), NO to begin search after the selected text.
 @result YES if found, NO if not found.
 */
- (BOOL)searchFor:(NSString *)string direction:(BOOL)forward caseSensitive:(BOOL)caseFlag wrap:(BOOL)wrapFlag startInSelection:(BOOL)startInSelection;
@end

@interface WebHTMLView (WebDocumentPrivateProtocols) <WebDocumentSelection, WebDocumentIncrementalSearching>
@end

#if TARGET_OS_IPHONE
@protocol WebPDFDocumentRepresentation <WebDocumentRepresentation>
/*!
    @method supportedMIMETypes
    @abstract Returns list of MIME types handled by this view.
    @result Array of strings representing the supported MIME types.
*/
+ (NSArray *)supportedMIMETypes;
+ (Class)_representationClassForWebFrame:(WebFrame *)webFrame;
@end

@protocol WebPDFDocumentView <WebDocumentView>
/*!
    @method supportedMIMETypes
    @abstract Returns list of MIME types handled by this view.
    @result Array of strings representing the supported MIME types.
*/
+ (NSArray *)supportedMIMETypes;
@end
#endif
