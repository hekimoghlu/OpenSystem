/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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
#import <WebKitLegacy/WebKitAvailability.h>

#if !TARGET_OS_IPHONE
#import <AppKit/AppKit.h>
#else
#import <WebKitLegacy/WAKAppKitStubs.h>
#import <WebKitLegacy/WAKView.h>
#endif

@class WebDataSource;
@class WebFrame;
@class WebFrameViewPrivate;

@protocol WebDocumentView;

/*!
    @class WebFrameView
*/
WEBKIT_CLASS_DEPRECATED_MAC(10_3, 10_14)
#if TARGET_OS_IPHONE
@interface WebFrameView : WAKView
#else
@interface WebFrameView : NSView
#endif
{
@package
    WebFrameViewPrivate *_private;
}

/*!
    @property webFrame
    @abstract The WebFrame associated with this WebFrameView
*/
@property (nonatomic, readonly, strong) WebFrame *webFrame;

/*!
    @property documentView
    @abstract The WebFrameView's document subview
    @discussion The subview that renders the WebFrameView's contents
*/
#if TARGET_OS_IPHONE
@property (nonatomic, readonly, strong) WAKView<WebDocumentView> *documentView;
#else
@property (nonatomic, readonly, strong) NSView<WebDocumentView> *documentView;
#endif

/*!
    @property allowsScrolling
    @abstract Whether the WebFrameView allows its document to be scrolled
*/
@property (nonatomic) BOOL allowsScrolling;

#if !TARGET_OS_IPHONE
/*!
    @property canPrintHeadersAndFooters
    @abstract Whether this frame can print headers and footers
*/
@property (nonatomic, readonly) BOOL canPrintHeadersAndFooters;

/*!
    @method printOperationWithPrintInfo
    @abstract Creates a print operation set up to print this frame
    @result A newly created print operation object
*/
- (NSPrintOperation *)printOperationWithPrintInfo:(NSPrintInfo *)printInfo;
#endif

/*!
    @property documentViewShouldHandlePrint
    @abstract Called by the host application before it initializes and runs a print operation.
    @discussion If NO is returned, the host application will abort its print operation and call -printDocumentView on the
    WebFrameView.  The document view is then expected to run its own print operation.  If YES is returned, the host 
    application's print operation will continue as normal.
*/
@property (nonatomic, readonly) BOOL documentViewShouldHandlePrint;

/*!
    @method printDocumentView
    @abstract Called by the host application when the WebFrameView returns YES from -documentViewShouldHandlePrint.
*/
- (void)printDocumentView;

@end
