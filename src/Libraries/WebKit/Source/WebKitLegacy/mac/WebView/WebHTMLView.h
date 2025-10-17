/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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

#if TARGET_OS_IPHONE
#import <WebKitLegacy/WAKAppKitStubs.h>
#import <WebKitLegacy/WAKView.h>
#endif

@class WebDataSource;
@class WebHTMLViewPrivate;

/*!
    @class WebHTMLView
    @discussion A document view of WebFrameView that displays HTML content.
    WebHTMLView is a NSControl because it hosts NSCells that are painted by WebCore's Aqua theme
    renderer (and those cells must be hosted by an enclosing NSControl in order to paint properly).
*/
#if !TARGET_OS_IPHONE
@interface WebHTMLView : NSControl <WebDocumentView, WebDocumentSearching>
#else
@interface WebHTMLView : WAKView <WebDocumentView, WebDocumentSearching>
#endif
{
@private
    WebHTMLViewPrivate *_private;
}

/*!
    @method setNeedsToApplyStyles:
    @abstract Sets flag to cause reapplication of style information.
    @param flag YES to apply style information, NO to not apply style information.
*/
- (void)setNeedsToApplyStyles:(BOOL)flag;

/*!
    @method reapplyStyles
    @discussion Immediately causes reapplication of style information to the view.  This should not be called directly,
    instead call setNeedsToApplyStyles:.
*/
- (void)reapplyStyles;

#if !TARGET_OS_IPHONE
- (void)outdent:(id)sender;
#endif

@end

