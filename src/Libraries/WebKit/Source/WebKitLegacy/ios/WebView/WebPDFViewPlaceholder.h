/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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
#import <WebKitLegacy/WebDocumentPrivate.h>

#if TARGET_OS_IPHONE

@class UIColor;
@class UIPDFDocument;
@protocol WebPDFViewPlaceholderDelegate;

/*!
    @class WebPDFViewPlaceholder
    @discussion This class represents a placeholder for PDFs. It is intended to allow main frame PDFs
    be drawn to the UI by some other object (ideally the delegate of this class) while still interfacing
    with WAK and WebKit correctly.
*/
@interface WebPDFViewPlaceholder : WAKView <WebPDFDocumentView, WebPDFDocumentRepresentation>

/*!
    @method setAsPDFDocRepAndView
    @abstract This methods sets [WebPDFViewPlaceholder class] as the document and view representations
    for PDF.
*/
+ (void)setAsPDFDocRepAndView;


/*!
 @property delegate
 @abstract A delegate object conforming to WebPDFViewPlaceholderDelegate that will be informed about various state changes.
 */
@property (weak) NSObject<WebPDFViewPlaceholderDelegate> *delegate;

/*!
 @property pageRects
 @abstract An array of CGRects (as NSValues) representing the bounds of each page in PDF document coordinate space.
 */
@property (readonly, retain) NSArray *pageRects;

/*!
 @property pageYOrigins
 @abstract An array of CGFloats (as NSNumbers) representing the minimum y for every page in the document.
 */
@property (readonly, retain) NSArray *pageYOrigins;

/*!
 @property document
 @abstract The CGPDFDocumentRef that this object represents. Until the document has loaded, this property will be NULL.
 */
@property (readonly) CGPDFDocumentRef document;
@property (readonly) CGPDFDocumentRef doc;

/*!
 @property totalPages
 @abstract Convenience access for the total number of pages in the wrapped document.
 */
@property (readonly) NSUInteger totalPages;

/*!
 @property title
 @abstract PDFs support a meta data field for the document's title. If this field is present in the PDF, title will be that string. 
 If not, title will be the file name.
 */
@property (readonly, retain) NSString *title;

/*!
 @property containerSize
 @abstract sets the size for the containing view. This is used to determine how big the shadows between pages should be.
 */
@property (assign) CGSize containerSize;

@property (nonatomic, readonly) BOOL didCompleteLayout;

- (void)clearDocument;

/*!
 @method didUnlockDocument
 @abstract Informs the PDF view placeholder that the PDF document has been unlocked. The result of this involves laying 
 out the pages, retaining the document title, and re-evaluating the document's javascript. This should be called on the WebThread.
 */
- (void)didUnlockDocument;

/*!
 @method rectForPageNumber
 @abstract Returns the PDF document coordinate space rect given a page number. pageNumber must be in the range [1,totalPages], 
 since page numbers are 1-based.
 */
- (CGRect)rectForPageNumber:(NSUInteger)pageNumber;

/*!
 @method simulateClickOnLinkToURL:
 @abstract This method simulates a user clicking on the passed in URL.
 */
- (void)simulateClickOnLinkToURL:(NSURL *)URL;

@end


/*!
    @protocol WebPDFViewPlaceholderDelegate
    @discussion This protocol is used to inform the object using the placeholder that the layout for the 
    document has been calculated.
*/
@protocol WebPDFViewPlaceholderDelegate

@optional

/*!
 @method viewWillClose
 @abstract This method is called to inform the delegate that the placeholder view's lifetime has ended. This might
 be called from either the main thread or the WebThread.
 */
- (void)viewWillClose;

/*!
    @method didCompleteLayout
    @abstract This method is called to inform the delegate that the placeholder has completed layout
    and determined the document's bounds. Will always be called on the main thread.
*/
- (void)didCompleteLayout;

@required

/*!
 @method cgPDFDocument
 @abstract The WebPDFViewPlaceholder may have abdicated responsibility for the underlying CGPDFDocument to the WebPDFViewPlaceholderDelegate.
 That means that there may be times when the document is needed, but the WebPDFViewPlaceholder no longer has a reference to it. In which case
 the WebPDFViewPlaceholderDelegate will be asked for the document.
 */
- (CGPDFDocumentRef)cgPDFDocument;

@end

#endif /* TARGET_OS_IPHONE */
