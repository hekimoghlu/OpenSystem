/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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

#import "APIResourceLoadClient.h"
#import "WKFoundation.h"
#import <wtf/CheckedPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakObjCPtr.h>

@class WKWebView;

@protocol _WKResourceLoadDelegate;

namespace API {
class ResourceLoadClient;
}

namespace WebKit {

class ResourceLoadDelegate : public CanMakeCheckedPtr<ResourceLoadDelegate> {
    WTF_MAKE_TZONE_ALLOCATED(ResourceLoadDelegate);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ResourceLoadDelegate);
public:
    explicit ResourceLoadDelegate(WKWebView *);
    ~ResourceLoadDelegate();

    std::unique_ptr<API::ResourceLoadClient> createResourceLoadClient();

    RetainPtr<id<_WKResourceLoadDelegate>> delegate();
    void setDelegate(id<_WKResourceLoadDelegate>);

private:
    class ResourceLoadClient : public API::ResourceLoadClient {
        WTF_MAKE_TZONE_ALLOCATED(ResourceLoadClient);
    public:
        explicit ResourceLoadClient(ResourceLoadDelegate&);
        ~ResourceLoadClient();

    private:
        // API::ResourceLoadClient
        void didSendRequest(ResourceLoadInfo&&, WebCore::ResourceRequest&&) const final;
        void didPerformHTTPRedirection(ResourceLoadInfo&&, WebCore::ResourceResponse&&, WebCore::ResourceRequest&&) const final;
        void didReceiveChallenge(ResourceLoadInfo&&, WebCore::AuthenticationChallenge&&) const final;
        void didReceiveResponse(ResourceLoadInfo&&, WebCore::ResourceResponse&&) const final;
        void didCompleteWithError(ResourceLoadInfo&&, WebCore::ResourceResponse&&, WebCore::ResourceError&&) const final;

        CheckedRef<ResourceLoadDelegate> m_resourceLoadDelegate;
    };

    WeakObjCPtr<WKWebView> m_webView;
    WeakObjCPtr<id <_WKResourceLoadDelegate> > m_delegate;

    struct {
        bool didSendRequest : 1;
        bool didPerformHTTPRedirection : 1;
        bool didReceiveChallenge : 1;
        bool didReceiveResponse : 1;
        bool didCompleteWithError : 1;
    } m_delegateMethods;
};

} // namespace WebKit
