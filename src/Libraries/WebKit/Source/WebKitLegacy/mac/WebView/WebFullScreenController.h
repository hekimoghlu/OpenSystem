/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 26, 2025.
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
#if ENABLE(FULLSCREEN_API) && !PLATFORM(IOS_FAMILY)

#import <WebCore/IntPoint.h>
#import <wtf/RefPtr.h>
#import <wtf/RetainPtr.h>

@class WebWindowFadeAnimation;
@class WebWindowScaleAnimation;
@class WebView;
namespace WebCore {
class Element;
class RenderBox;
class EventListener;
}

@interface WebFullScreenController : NSWindowController {
@private
    RefPtr<WebCore::Element> _element;
    WebView *_webView;
    RetainPtr<NSView> _webViewPlaceholder;
    RetainPtr<WebWindowScaleAnimation> _scaleAnimation;
    RetainPtr<WebWindowFadeAnimation> _fadeAnimation;
    RetainPtr<NSWindow> _backgroundWindow;
    NSRect _initialFrame;
    NSRect _finalFrame;
    WebCore::IntPoint _scrollPosition;
    float _savedScale;

    BOOL _isEnteringFullScreen;
    BOOL _isExitingFullScreen;
    BOOL _isFullScreen;
}

@property (readonly) NSRect initialFrame;
@property (readonly) NSRect finalFrame;

- (WebView*)webView;
- (void)setWebView:(WebView*)webView;

- (NSView*)webViewPlaceholder;

- (BOOL)isFullScreen;

- (void)setElement:(RefPtr<WebCore::Element>&&)element;
- (WebCore::Element*)element;

- (void)enterFullScreen:(NSScreen *)screen;
- (void)exitFullScreen;
- (void)close;
@end

#endif // ENABLE(FULLSCREEN_API) && !PLATFORM(IOS_FAMILY)
