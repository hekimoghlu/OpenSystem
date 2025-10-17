/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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
#import "ResourceLoadDelegate.h"

#import <WebCore/AuthenticationMac.h>
#import "AuthenticationChallengeProxy.h"
#import "WKNSURLAuthenticationChallenge.h"
#import "_WKResourceLoadDelegate.h"
#import "_WKResourceLoadInfoInternal.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ResourceLoadDelegate);

ResourceLoadDelegate::ResourceLoadDelegate(WKWebView *webView)
    : m_webView(webView)
{
}

ResourceLoadDelegate::~ResourceLoadDelegate() = default;

std::unique_ptr<API::ResourceLoadClient> ResourceLoadDelegate::createResourceLoadClient()
{
    return makeUnique<ResourceLoadClient>(*this);
}

RetainPtr<id<_WKResourceLoadDelegate>> ResourceLoadDelegate::delegate()
{
    return m_delegate.get();
}

void ResourceLoadDelegate::setDelegate(id <_WKResourceLoadDelegate> delegate)
{
    m_delegate = delegate;

    // resourceWithID:
    // type:
    // _WKFrameHandle frame:
    // _WKFrameHandle parentFrame:
    
    m_delegateMethods.didSendRequest = [delegate respondsToSelector:@selector(webView:resourceLoad:didSendRequest:)];
    m_delegateMethods.didPerformHTTPRedirection = [delegate respondsToSelector:@selector(webView:resourceLoad:didPerformHTTPRedirection:newRequest:)];
    m_delegateMethods.didReceiveChallenge = [delegate respondsToSelector:@selector(webView:resourceLoad:didReceiveChallenge:)];
    m_delegateMethods.didReceiveResponse = [delegate respondsToSelector:@selector(webView:resourceLoad:didReceiveResponse:)];
    m_delegateMethods.didCompleteWithError = [delegate respondsToSelector:@selector(webView:resourceLoad:didCompleteWithError:response:)];
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(ResourceLoadDelegate::ResourceLoadClient);

ResourceLoadDelegate::ResourceLoadClient::ResourceLoadClient(ResourceLoadDelegate& delegate)
    : m_resourceLoadDelegate(delegate)
{
}

ResourceLoadDelegate::ResourceLoadClient::~ResourceLoadClient() = default;

void ResourceLoadDelegate::ResourceLoadClient::didSendRequest(WebKit::ResourceLoadInfo&& loadInfo, WebCore::ResourceRequest&& request) const
{
    if (!m_resourceLoadDelegate->m_delegateMethods.didSendRequest)
        return;

    auto delegate = m_resourceLoadDelegate->m_delegate.get();
    if (!delegate)
        return;

    [delegate webView:m_resourceLoadDelegate->m_webView.get().get() resourceLoad:wrapper(API::ResourceLoadInfo::create(WTFMove(loadInfo)).get()) didSendRequest:request.nsURLRequest(WebCore::HTTPBodyUpdatePolicy::UpdateHTTPBody)];
}

void ResourceLoadDelegate::ResourceLoadClient::didPerformHTTPRedirection(WebKit::ResourceLoadInfo&& loadInfo, WebCore::ResourceResponse&& response, WebCore::ResourceRequest&& request) const
{
    if (!m_resourceLoadDelegate->m_delegateMethods.didPerformHTTPRedirection)
        return;

    auto delegate = m_resourceLoadDelegate->m_delegate.get();
    if (!delegate)
        return;

    [delegate webView:m_resourceLoadDelegate->m_webView.get().get() resourceLoad:wrapper(API::ResourceLoadInfo::create(WTFMove(loadInfo)).get()) didPerformHTTPRedirection:response.nsURLResponse() newRequest:request.nsURLRequest(WebCore::HTTPBodyUpdatePolicy::DoNotUpdateHTTPBody)];
}

void ResourceLoadDelegate::ResourceLoadClient::didReceiveChallenge(WebKit::ResourceLoadInfo&& loadInfo, WebCore::AuthenticationChallenge&& challenge) const
{
    if (!m_resourceLoadDelegate->m_delegateMethods.didReceiveChallenge)
        return;

    auto delegate = m_resourceLoadDelegate->m_delegate.get();
    if (!delegate)
        return;

    [delegate webView:m_resourceLoadDelegate->m_webView.get().get() resourceLoad:wrapper(API::ResourceLoadInfo::create(WTFMove(loadInfo)).get()) didReceiveChallenge:mac(challenge)];
}

void ResourceLoadDelegate::ResourceLoadClient::didReceiveResponse(WebKit::ResourceLoadInfo&& loadInfo, WebCore::ResourceResponse&& response) const
{
    if (!m_resourceLoadDelegate->m_delegateMethods.didReceiveResponse)
        return;

    auto delegate = m_resourceLoadDelegate->m_delegate.get();
    if (!delegate)
        return;

    [delegate webView:m_resourceLoadDelegate->m_webView.get().get() resourceLoad:wrapper(API::ResourceLoadInfo::create(WTFMove(loadInfo)).get()) didReceiveResponse:response.nsURLResponse()];
}

void ResourceLoadDelegate::ResourceLoadClient::didCompleteWithError(WebKit::ResourceLoadInfo&& loadInfo, WebCore::ResourceResponse&& response, WebCore::ResourceError&& error) const
{
    if (!m_resourceLoadDelegate->m_delegateMethods.didCompleteWithError)
        return;

    auto delegate = m_resourceLoadDelegate->m_delegate.get();
    if (!delegate)
        return;

    [delegate webView:m_resourceLoadDelegate->m_webView.get().get() resourceLoad:wrapper(API::ResourceLoadInfo::create(WTFMove(loadInfo)).get()) didCompleteWithError:error.nsError() response:response.nsURLResponse()];
}

} // namespace WebKit
