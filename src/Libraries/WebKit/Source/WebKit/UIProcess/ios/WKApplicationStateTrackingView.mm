/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 22, 2022.
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
#import "config.h"
#import "WKApplicationStateTrackingView.h"

#if PLATFORM(IOS_FAMILY)

#import "ApplicationStateTracker.h"
#import "Logging.h"
#import "WKWebViewInternal.h"
#import "WebPageProxy.h"
#import <wtf/RetainPtr.h>
#import <wtf/WeakObjCPtr.h>

@implementation WKApplicationStateTrackingView {
    WeakObjCPtr<WKWebView> _webViewToTrack;
    RefPtr<WebKit::ApplicationStateTracker> _applicationStateTracker;
}

- (instancetype)initWithFrame:(CGRect)frame webView:(WKWebView *)webView
{
    if (!(self = [super initWithFrame:frame]))
        return nil;

    _webViewToTrack = webView;
    _applicationStateTracker = WebKit::ApplicationStateTracker::create(self, @selector(_applicationDidEnterBackground), @selector(_applicationWillEnterForeground), @selector(_willBeginSnapshotSequence), @selector(_didCompleteSnapshotSequence));
    return self;
}

- (void)willMoveToWindow:(UIWindow *)newWindow
{
    if ((self.window && !self._contentView.window) || newWindow)
        return;

    auto page = [_webViewToTrack _page];
    RELEASE_LOG(ViewState, "%p - WKApplicationStateTrackingView: View with page [%p, pageProxyID=%" PRIu64 "] was removed from a window, _lastObservedStateWasBackground=%d", self, page.get(), page ? page->identifier().toUInt64() : 0, page ? page->lastObservedStateWasBackground() : false);
    _applicationStateTracker->setWindow(nil);
}

- (void)didMoveToWindow
{
    if (!self._contentView.window)
        return;

    auto page = [_webViewToTrack _page];
    bool lastObservedStateWasBackground = page ? page->lastObservedStateWasBackground() : false;

    _applicationStateTracker->setWindow(self._contentView.window);
    RELEASE_LOG(ViewState, "%p - WKApplicationStateTrackingView: View with page [%p, pageProxyID=%" PRIu64 "] was added to a window, _lastObservedStateWasBackground=%d, isNowBackground=%d", self, page.get(), page ? page->identifier().toUInt64() : 0, lastObservedStateWasBackground, [self isBackground]);

    if (lastObservedStateWasBackground && ![self isBackground])
        [self _applicationWillEnterForeground];
    else if (!lastObservedStateWasBackground && [self isBackground])
        [self _applicationDidEnterBackground];
}

- (void)_applicationDidEnterBackground
{
    auto page = [_webViewToTrack _page];
    if (!page)
        return;

    page->applicationDidEnterBackground();
    page->activityStateDidChange(WebCore::allActivityStates() - WebCore::ActivityState::IsInWindow);
}

- (void)_applicationDidFinishSnapshottingAfterEnteringBackground
{
    auto page = [_webViewToTrack _page];
    if (!page)
        return;

    RELEASE_LOG(ViewState, "%p - WKApplicationStateTrackingView: View with page [%p, pageProxyID=%" PRIu64 "] did finish snapshotting after entering background", self, page.get(), page ? page->identifier().toUInt64() : 0);
    page->applicationDidFinishSnapshottingAfterEnteringBackground();
}

- (void)_applicationWillEnterForeground
{
    auto page = [_webViewToTrack _page];
    if (!page)
        return;

    page->applicationWillEnterForeground();
    page->activityStateDidChange(WebCore::allActivityStates() - WebCore::ActivityState::IsInWindow, WebKit::WebPageProxy::ActivityStateChangeDispatchMode::Immediate, WebKit::WebPageProxy::ActivityStateChangeReplyMode::Synchronous);
}

- (void)_willBeginSnapshotSequence
{
    auto page = [_webViewToTrack _page];
    if (!page)
        return;

    if (!self._contentView.window)
        return;

    RELEASE_LOG(ViewState, "%p - WKApplicationStateTrackingView: View with page [%p, pageProxyID=%" PRIu64 "] will begin snapshot sequence", self, page.get(), page ? page->identifier().toUInt64() : 0);
    page->setIsTakingSnapshotsForApplicationSuspension(true);
}

- (void)_didCompleteSnapshotSequence
{
    auto page = [_webViewToTrack _page];
    if (!page)
        return;

    if (!self._contentView.window)
        return;

    RELEASE_LOG(ViewState, "%p - WKApplicationStateTrackingView: View with page [%p, pageProxyID=%" PRIu64 "] did complete snapshot sequence", self, page.get(), page ? page->identifier().toUInt64() : 0);

    page->setIsTakingSnapshotsForApplicationSuspension(false);
    if ([self isBackground])
        page->applicationDidFinishSnapshottingAfterEnteringBackground();
}

- (BOOL)isBackground
{
    return _applicationStateTracker ? _applicationStateTracker->isInBackground() : YES;
}

- (UIView *)_contentView
{
    return self;
}

@end

#endif // PLATFORM(IOS_FAMILY)
