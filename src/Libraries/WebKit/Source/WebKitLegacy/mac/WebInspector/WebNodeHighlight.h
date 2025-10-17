/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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
#if PLATFORM(IOS_FAMILY)
#import <QuartzCore/CALayer.h>
#import <WebKitLegacy/WAKAppKitStubs.h>
#import <WebKitLegacy/WAKView.h>
#endif
#import <wtf/NakedPtr.h>

@class WebNodeHighlightView;
#if PLATFORM(IOS_FAMILY)
@class WebView;
#endif

namespace WebCore {
class InspectorController;
}

#if PLATFORM(IOS_FAMILY)
@interface WebHighlightLayer : CALayer {
    WebNodeHighlightView *_view;
    WebView *_webView;
}
- (id)initWithHighlightView:(WebNodeHighlightView *)view webView:(WebView *)webView;
@end
#endif

@interface WebNodeHighlight : NSObject {
    NSView *_targetView;
#if !PLATFORM(IOS_FAMILY)
    NSWindow *_highlightWindow;
#else
    WebHighlightLayer *_highlightLayer;
#endif
    WebNodeHighlightView *_highlightView;
    NakedPtr<WebCore::InspectorController> _inspectorController;
    id _delegate;
}
- (id)initWithTargetView:(NSView *)targetView inspectorController:(NakedPtr<WebCore::InspectorController>)inspectorController;

- (void)setDelegate:(id)delegate;
- (id)delegate;

- (void)attach;
- (void)detach;

- (NSView *)targetView;
- (WebNodeHighlightView *)highlightView;

- (NakedPtr<WebCore::InspectorController>)inspectorController;

#if !PLATFORM(IOS_FAMILY)
- (void)setNeedsUpdateInTargetViewRect:(NSRect)rect;
#else
- (void)setNeedsDisplay;
#endif
@end

@interface NSObject (WebNodeHighlightDelegate)
- (void)didAttachWebNodeHighlight:(WebNodeHighlight *)highlight;
- (void)willDetachWebNodeHighlight:(WebNodeHighlight *)highlight;
@end
