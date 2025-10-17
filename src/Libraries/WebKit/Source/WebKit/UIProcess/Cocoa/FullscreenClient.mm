/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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
#import "FullscreenClient.h"

#import "WKWebViewInternal.h"
#import "_WKFullscreenDelegate.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FullscreenClient);

FullscreenClient::FullscreenClient(WKWebView *webView)
    : m_webView(webView)
{
}

RetainPtr<id <_WKFullscreenDelegate>> FullscreenClient::delegate()
{
    return m_delegate.get();
}

void FullscreenClient::setDelegate(id <_WKFullscreenDelegate> delegate)
{
    m_delegate = delegate;

#if PLATFORM(MAC)
    m_delegateMethods.webViewWillEnterFullscreen = [delegate respondsToSelector:@selector(_webViewWillEnterFullscreen:)];
    m_delegateMethods.webViewDidEnterFullscreen = [delegate respondsToSelector:@selector(_webViewDidEnterFullscreen:)];
    m_delegateMethods.webViewWillExitFullscreen = [delegate respondsToSelector:@selector(_webViewWillExitFullscreen:)];
    m_delegateMethods.webViewDidExitFullscreen = [delegate respondsToSelector:@selector(_webViewDidExitFullscreen:)];
#else
    m_delegateMethods.webViewWillEnterElementFullscreen = [delegate respondsToSelector:@selector(_webViewWillEnterElementFullscreen:)];
    m_delegateMethods.webViewDidEnterElementFullscreen = [delegate respondsToSelector:@selector(_webViewDidEnterElementFullscreen:)];
    m_delegateMethods.webViewWillExitElementFullscreen = [delegate respondsToSelector:@selector(_webViewWillExitElementFullscreen:)];
    m_delegateMethods.webViewDidExitElementFullscreen = [delegate respondsToSelector:@selector(_webViewDidExitElementFullscreen:)];
    m_delegateMethods.webViewRequestPresentingViewController = [delegate respondsToSelector:@selector(_webView:requestPresentingViewControllerWithCompletionHandler:)];
#endif
#if ENABLE(QUICKLOOK_FULLSCREEN)
    m_delegateMethods.webViewDidFullscreenImageWithQuickLook = [delegate respondsToSelector:@selector(_webView:didFullscreenImageWithQuickLook:)];
#endif
}

void FullscreenClient::willEnterFullscreen(WebPageProxy*)
{
    [m_webView willChangeValueForKey:@"fullscreenState"];
    [m_webView didChangeValueForKey:@"fullscreenState"];
#if PLATFORM(MAC)
    if (m_delegateMethods.webViewWillEnterFullscreen)
        [m_delegate.get() _webViewWillEnterFullscreen:m_webView];
#else
    if (m_delegateMethods.webViewWillEnterElementFullscreen)
        [m_delegate.get() _webViewWillEnterElementFullscreen:m_webView];
#endif
}

void FullscreenClient::didEnterFullscreen(WebPageProxy*)
{
    [m_webView willChangeValueForKey:@"fullscreenState"];
    [m_webView didChangeValueForKey:@"fullscreenState"];
#if PLATFORM(MAC)
    if (m_delegateMethods.webViewDidEnterFullscreen)
        [m_delegate.get() _webViewDidEnterFullscreen:m_webView];
#else
    if (m_delegateMethods.webViewDidEnterElementFullscreen)
        [m_delegate.get() _webViewDidEnterElementFullscreen:m_webView];
#endif

#if ENABLE(QUICKLOOK_FULLSCREEN)
    if (auto fullScreenController = [m_webView fullScreenWindowController]) {
        CGSize imageDimensions = fullScreenController.imageDimensions;
        if (fullScreenController.isUsingQuickLook && m_delegateMethods.webViewDidFullscreenImageWithQuickLook)
            [m_delegate.get() _webView:m_webView didFullscreenImageWithQuickLook:imageDimensions];
    }
#endif // ENABLE(QUICKLOOK_FULLSCREEN)
}

void FullscreenClient::willExitFullscreen(WebPageProxy*)
{
    [m_webView willChangeValueForKey:@"fullscreenState"];
    [m_webView didChangeValueForKey:@"fullscreenState"];
#if PLATFORM(MAC)
    if (m_delegateMethods.webViewWillExitFullscreen)
        [m_delegate.get() _webViewWillExitFullscreen:m_webView];
#else
    if (m_delegateMethods.webViewWillExitElementFullscreen)
        [m_delegate.get() _webViewWillExitElementFullscreen:m_webView];
#endif
}

void FullscreenClient::didExitFullscreen(WebPageProxy*)
{
    [m_webView willChangeValueForKey:@"fullscreenState"];
    [m_webView didChangeValueForKey:@"fullscreenState"];
#if PLATFORM(MAC)
    if (m_delegateMethods.webViewDidExitFullscreen)
        [m_delegate.get() _webViewDidExitFullscreen:m_webView];
#else
    if (m_delegateMethods.webViewDidExitElementFullscreen)
        [m_delegate.get() _webViewDidExitElementFullscreen:m_webView];
#endif
}

#if PLATFORM(IOS_FAMILY)
void FullscreenClient::requestPresentingViewController(CompletionHandler<void(UIViewController *, NSError *)>&& completionHandler)
{
    if (!m_delegateMethods.webViewRequestPresentingViewController)
        return completionHandler(nil, nil);

    [m_delegate _webView:m_webView requestPresentingViewControllerWithCompletionHandler:makeBlockPtr(WTFMove(completionHandler)).get()];
}
#endif

} // namespace WebKit
