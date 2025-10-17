/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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
#import "_WKDataTaskInternal.h"

#import "APIDataTask.h"
#import "APIDataTaskClient.h"
#import "AuthenticationChallengeDispositionCocoa.h"
#import "CompletionHandlerCallChecker.h"
#import "WebPageProxy.h"
#import "_WKDataTaskDelegate.h"
#import <WebCore/AuthenticationMac.h>
#import <WebCore/Credential.h>
#import <WebCore/ResourceRequest.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/BlockPtr.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/cocoa/SpanCocoa.h>

class WKDataTaskClient final : public API::DataTaskClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WKDataTaskClient);
public:
    static Ref<WKDataTaskClient> create(id <_WKDataTaskDelegate> delegate) { return adoptRef(*new WKDataTaskClient(delegate)); }
private:
    explicit WKDataTaskClient(id <_WKDataTaskDelegate> delegate)
        : m_delegate(delegate)
        , m_respondsToDidReceiveAuthenticationChallenge([delegate respondsToSelector:@selector(dataTask:didReceiveAuthenticationChallenge:completionHandler:)])
        , m_respondsToWillPerformHTTPRedirection([delegate respondsToSelector:@selector(dataTask:willPerformHTTPRedirection:newRequest:decisionHandler:)])
        , m_respondsToDidReceiveResponse([delegate respondsToSelector:@selector(dataTask:didReceiveResponse:decisionHandler:)])
        , m_respondsToDidReceiveData([delegate respondsToSelector:@selector(dataTask:didReceiveData:)])
        , m_respondsToDidCompleteWithError([delegate respondsToSelector:@selector(dataTask:didCompleteWithError:)]) { }

    void didReceiveChallenge(API::DataTask& task, WebCore::AuthenticationChallenge&& challenge, CompletionHandler<void(WebKit::AuthenticationChallengeDisposition, WebCore::Credential&&)>&& completionHandler) const final
    {
        if (!m_delegate || !m_respondsToDidReceiveAuthenticationChallenge)
            return completionHandler(WebKit::AuthenticationChallengeDisposition::RejectProtectionSpaceAndContinue, { });
        auto checker = WebKit::CompletionHandlerCallChecker::create(m_delegate.get().get(), @selector(dataTask:didReceiveAuthenticationChallenge:completionHandler:));
        [m_delegate dataTask:wrapper(task) didReceiveAuthenticationChallenge:mac(challenge) completionHandler:makeBlockPtr([checker = WTFMove(checker), completionHandler = WTFMove(completionHandler)](NSURLSessionAuthChallengeDisposition disposition, NSURLCredential *credential) mutable {
            if (checker->completionHandlerHasBeenCalled())
                return;
            checker->didCallCompletionHandler();
            completionHandler(WebKit::toAuthenticationChallengeDisposition(disposition), WebCore::Credential(credential));
        }).get()];
    }

    void willPerformHTTPRedirection(API::DataTask& task, WebCore::ResourceResponse&& response, WebCore::ResourceRequest&& request, CompletionHandler<void(bool)>&& completionHandler) const final
    {
        if (!m_delegate || !m_respondsToWillPerformHTTPRedirection)
            return completionHandler(true);
        auto checker = WebKit::CompletionHandlerCallChecker::create(m_delegate.get().get(), @selector(dataTask:willPerformHTTPRedirection:newRequest:decisionHandler:));
        [m_delegate dataTask:wrapper(task) willPerformHTTPRedirection:(NSHTTPURLResponse *)response.nsURLResponse() newRequest:request.nsURLRequest(WebCore::HTTPBodyUpdatePolicy::UpdateHTTPBody) decisionHandler:makeBlockPtr([checker = WTFMove(checker), completionHandler = WTFMove(completionHandler)] (_WKDataTaskRedirectPolicy policy) mutable {
            if (checker->completionHandlerHasBeenCalled())
                return;
            checker->didCallCompletionHandler();
            completionHandler(policy == _WKDataTaskRedirectPolicyAllow);
        }).get()];
    }

    void didReceiveResponse(API::DataTask& task, WebCore::ResourceResponse&& response, CompletionHandler<void(bool)>&& completionHandler) const final
    {
        if (!m_delegate || !m_respondsToDidReceiveResponse)
            return completionHandler(true);
        auto checker = WebKit::CompletionHandlerCallChecker::create(m_delegate.get().get(), @selector(dataTask:didReceiveResponse:decisionHandler:));
        [m_delegate dataTask:wrapper(task) didReceiveResponse:response.nsURLResponse() decisionHandler:makeBlockPtr([checker = WTFMove(checker), completionHandler = WTFMove(completionHandler)] (_WKDataTaskResponsePolicy policy) mutable {
            if (checker->completionHandlerHasBeenCalled())
                return;
            checker->didCallCompletionHandler();
            completionHandler(policy == _WKDataTaskResponsePolicyAllow);
        }).get()];
    }

    void didReceiveData(API::DataTask& task, std::span<const uint8_t> data) const final
    {
        if (!m_delegate || !m_respondsToDidReceiveData)
            return;
        [m_delegate dataTask:wrapper(task) didReceiveData:toNSData(data).get()];
    }

    void didCompleteWithError(API::DataTask& task, WebCore::ResourceError&& error) const final
    {
        if (!m_delegate || !m_respondsToDidCompleteWithError)
            return;
        [m_delegate dataTask:wrapper(task) didCompleteWithError:error.nsError()];
        wrapper(task)->_delegate = nil;
    }

    WeakObjCPtr<id <_WKDataTaskDelegate> > m_delegate;

    bool m_respondsToDidReceiveAuthenticationChallenge : 1;
    bool m_respondsToWillPerformHTTPRedirection : 1;
    bool m_respondsToDidReceiveResponse : 1;
    bool m_respondsToDidReceiveData : 1;
    bool m_respondsToDidCompleteWithError : 1;
};

@implementation _WKDataTask

- (void)cancel
{
    _dataTask->cancel();
    _delegate = nil;
}

- (WKWebView *)webView
{
    auto* page = _dataTask->page();
    if (!page)
        return nil;
    return page->cocoaView().get();
}

- (id <_WKDataTaskDelegate>)delegate
{
    return _delegate.get();
}

- (void)setDelegate:(id <_WKDataTaskDelegate>)delegate
{
    _delegate = delegate;
    _dataTask->setClient(WKDataTaskClient::create(delegate));
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKDataTask.class, self))
        return;
    _dataTask->~DataTask();
    [super dealloc];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_dataTask;
}

@end
