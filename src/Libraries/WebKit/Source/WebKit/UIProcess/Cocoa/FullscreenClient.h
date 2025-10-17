/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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

#import "WKFoundation.h"

#import "APIFullscreenClient.h"
#import "WKWebView.h"
#import <wtf/RetainPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakObjCPtr.h>

@protocol _WKFullscreenDelegate;

namespace WebKit {

class FullscreenClient final : public API::FullscreenClient {
    WTF_MAKE_TZONE_ALLOCATED(FullscreenClient);
public:
    explicit FullscreenClient(WKWebView *);
    ~FullscreenClient() { };

    bool isType(API::FullscreenClient::Type target) const final { return target == API::FullscreenClient::WebKitType; };

    RetainPtr<id<_WKFullscreenDelegate>> delegate();
    void setDelegate(id<_WKFullscreenDelegate>);

    void willEnterFullscreen(WebPageProxy*) final;
    void didEnterFullscreen(WebPageProxy*) final;
    void willExitFullscreen(WebPageProxy*) final;
    void didExitFullscreen(WebPageProxy*) final;

#if PLATFORM(IOS_FAMILY)
    void requestPresentingViewController(CompletionHandler<void(UIViewController *, NSError *)>&&) final;
#endif

private:
    WKWebView *m_webView;
    WeakObjCPtr<id <_WKFullscreenDelegate> > m_delegate;

    struct {
#if PLATFORM(MAC)
        bool webViewWillEnterFullscreen : 1 { false };
        bool webViewDidEnterFullscreen : 1 { false };
        bool webViewWillExitFullscreen : 1 { false };
        bool webViewDidExitFullscreen : 1 { false };
#else
        bool webViewWillEnterElementFullscreen : 1 { false };
        bool webViewDidEnterElementFullscreen : 1 { false };
        bool webViewWillExitElementFullscreen : 1 { false };
        bool webViewDidExitElementFullscreen : 1 { false };
#endif
#if ENABLE(QUICKLOOK_FULLSCREEN)
        bool webViewDidFullscreenImageWithQuickLook : 1 { false };
#endif
#if PLATFORM(IOS_FAMILY)
        bool webViewRequestPresentingViewController : 1 { false };
#endif
    } m_delegateMethods;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::FullscreenClient) \
static bool isType(const API::FullscreenClient& client) { return client.isType(API::FullscreenClient::WebKitType); } \
SPECIALIZE_TYPE_TRAITS_END()
