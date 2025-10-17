/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 21, 2022.
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
#import <WebKitLegacy/WebHTMLView.h>
#import <WebKitLegacy/WebViewPrivate.h>

#if PLATFORM(IOS_FAMILY)
#if !defined(IBAction)
#define IBAction void
#endif
#endif

/*!
@protocol _WebDocumentZooming
@discussion Optional protocol for a view that wants to handle its own zoom.
*/
@protocol _WebDocumentZooming <NSObject>

// Methods to perform the actual commands
- (IBAction)_zoomOut:(id)sender;
- (IBAction)_zoomIn:(id)sender;
- (IBAction)_resetZoom:(id)sender;

// Whether or not the commands can be executed.
- (BOOL)_canZoomOut;
- (BOOL)_canZoomIn;
- (BOOL)_canResetZoom;

@end

@protocol WebDocumentElement <NSObject>
- (NSDictionary *)elementAtPoint:(NSPoint)point;
- (NSDictionary *)elementAtPoint:(NSPoint)point allowShadowContent:(BOOL)allow;
@end

@protocol WebMultipleTextMatches <NSObject>
- (void)setMarkedTextMatchesAreHighlighted:(BOOL)newValue;
- (BOOL)markedTextMatchesAreHighlighted;
- (NSUInteger)countMatchesForText:(NSString *)string inDOMRange:(DOMRange *)range options:(WebFindOptions)options limit:(NSUInteger)limit markMatches:(BOOL)markMatches;
- (void)unmarkAllTextMatches;
- (NSArray *)rectsForTextMatches;
@end

@protocol WebDocumentOptionsSearching <NSObject>
// Prefixed with an underscore to avoid conflict with Mail's -[WebHTMLView(MailExtras) findString:options:].
- (BOOL)_findString:(NSString *)string options:(WebFindOptions)options;
@end

/* Used to save and restore state in the view, typically when going back/forward */
@protocol _WebDocumentViewState <NSObject>
- (NSPoint)scrollPoint;
- (void)setScrollPoint:(NSPoint)p;
- (id)viewState;
- (void)setViewState:(id)statePList;
@end

@interface WebHTMLView (WebDocumentInternalProtocols) <WebDocumentElement, WebMultipleTextMatches, WebDocumentOptionsSearching>
@end
