/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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
#import "SubFrameSOAuthorizationSession.h"

#if HAVE(APP_SSO)

#import "APIFrameHandle.h"
#import "APINavigationAction.h"
#import "WebFrameProxy.h"
#import "WebPageProxy.h"
#import "WebProcessProxy.h"
#import <WebCore/ContentSecurityPolicy.h>
#import <WebCore/HTTPParsers.h>
#import <WebCore/HTTPStatusCodes.h>
#import <WebCore/ResourceResponse.h>
#import <wtf/RunLoop.h>
#import <wtf/cocoa/VectorCocoa.h>
#import <wtf/text/MakeString.h>

namespace WebKit {
using namespace WebCore;

#define AUTHORIZATIONSESSION_RELEASE_LOG(fmt, ...) RELEASE_LOG(AppSSO, "%p - [InitiatingAction=%s][State=%s] SubFrameSOAuthorizationSession::" fmt, this, initiatingActionString().characters(), stateString().characters(), ##__VA_ARGS__)

namespace {

constexpr auto soAuthorizationPostDidStartMessageToParent = "<script>parent.postMessage('SOAuthorizationDidStart', '*');</script>"_s;
constexpr auto soAuthorizationPostDidCancelMessageToParent = "<script>parent.postMessage('SOAuthorizationDidCancel', '*');</script>"_s;

} // namespace

Ref<SOAuthorizationSession> SubFrameSOAuthorizationSession::create(RetainPtr<WKSOAuthorizationDelegate> delegate, Ref<API::NavigationAction>&& navigationAction, WebPageProxy& page, Callback&& completionHandler, std::optional<FrameIdentifier> frameID)
{
    return adoptRef(*new SubFrameSOAuthorizationSession(delegate, WTFMove(navigationAction), page, WTFMove(completionHandler), frameID));
}

SubFrameSOAuthorizationSession::SubFrameSOAuthorizationSession(RetainPtr<WKSOAuthorizationDelegate> delegate, Ref<API::NavigationAction>&& navigationAction, WebPageProxy& page, Callback&& completionHandler, std::optional<FrameIdentifier> frameID)
    : NavigationSOAuthorizationSession(delegate, WTFMove(navigationAction), page, InitiatingAction::SubFrame, WTFMove(completionHandler))
    , m_frameID(frameID)
{
    if (RefPtr frame = WebFrameProxy::webFrame(m_frameID))
        frame->frameLoadState().addObserver(*this);
}

SubFrameSOAuthorizationSession::~SubFrameSOAuthorizationSession()
{
    if (RefPtr frame = WebFrameProxy::webFrame(m_frameID))
        frame->frameLoadState().removeObserver(*this);
}

void SubFrameSOAuthorizationSession::fallBackToWebPathInternal()
{
    AUTHORIZATIONSESSION_RELEASE_LOG("fallBackToWebPathInternal: navigationAction=%p", navigationAction());
    ASSERT(navigationAction());
    appendRequestToLoad(URL(navigationAction()->request().url()), Vector<uint8_t>(unsafeSpan8(soAuthorizationPostDidCancelMessageToParent)));
    appendRequestToLoad(URL(navigationAction()->request().url()), String(navigationAction()->request().httpReferrer()));
}

void SubFrameSOAuthorizationSession::abortInternal()
{
    AUTHORIZATIONSESSION_RELEASE_LOG("abortInternal");
    fallBackToWebPathInternal();
}

void SubFrameSOAuthorizationSession::completeInternal(const WebCore::ResourceResponse& response, NSData *data)
{
    AUTHORIZATIONSESSION_RELEASE_LOG("completeInternal: httpState=%d", response.httpStatusCode());
    if (response.httpStatusCode() != httpStatus200OK) {
        fallBackToWebPathInternal();
        return;
    }
    appendRequestToLoad(URL(response.url()), makeVector(data));
}

void SubFrameSOAuthorizationSession::beforeStart()
{
    AUTHORIZATIONSESSION_RELEASE_LOG("beforeStart");
    // Cancelled the current load before loading the data to post SOAuthorizationDidStart to the parent frame.
    invokeCallback(true);
    ASSERT(navigationAction());
    appendRequestToLoad(URL(navigationAction()->request().url()), Vector<uint8_t>(unsafeSpan8(soAuthorizationPostDidStartMessageToParent)));
}

void SubFrameSOAuthorizationSession::didFinishLoad(IsMainFrame, const URL&)
{
    AUTHORIZATIONSESSION_RELEASE_LOG("didFinishLoad");
    RefPtr frame = WebFrameProxy::webFrame(m_frameID);
    ASSERT(frame);
    if (m_requestsToLoad.isEmpty() || m_requestsToLoad.first().first != frame->url())
        return;
    m_requestsToLoad.takeFirst();
    loadRequestToFrame();
}

void SubFrameSOAuthorizationSession::appendRequestToLoad(URL&& url, Supplement&& supplement)
{
    m_requestsToLoad.append({ WTFMove(url), WTFMove(supplement) });
    if (m_requestsToLoad.size() == 1)
        loadRequestToFrame();
}

void SubFrameSOAuthorizationSession::loadRequestToFrame()
{
    AUTHORIZATIONSESSION_RELEASE_LOG("loadRequestToFrame");
    RefPtr page = this->page();
    if (!page || m_requestsToLoad.isEmpty())
        return;

    if (RefPtr frame = WebFrameProxy::webFrame(m_frameID)) {
        page->setShouldSuppressSOAuthorizationInNextNavigationPolicyDecision();
        auto& url = m_requestsToLoad.first().first;
        WTF::switchOn(m_requestsToLoad.first().second, [&](const Vector<uint8_t>& data) {
            frame->loadData(data, "text/html"_s, "UTF-8"_s, url);
        }, [&](const String& referrer) {
            frame->loadURL(url, referrer);
        });
    }
}

bool SubFrameSOAuthorizationSession::shouldInterruptLoadForXFrameOptions(Vector<Ref<SecurityOrigin>>&& frameAncestorOrigins, const String& xFrameOptions, const URL& url)
{
    switch (parseXFrameOptionsHeader(xFrameOptions)) {
    case XFrameOptionsDisposition::None:
    case XFrameOptionsDisposition::AllowAll:
        return false;
    case XFrameOptionsDisposition::Deny:
        return true;
    case XFrameOptionsDisposition::SameOrigin: {
        auto origin = SecurityOrigin::create(url);
        for (auto& ancestorOrigin : frameAncestorOrigins) {
            if (!origin->isSameSchemeHostPort(ancestorOrigin))
                return true;
        }
        return false;
    }
    case XFrameOptionsDisposition::Conflict: {
        auto errorMessage = makeString("Multiple 'X-Frame-Options' headers with conflicting values ('"_s, xFrameOptions, "') encountered. Falling back to 'DENY'."_s);
        AUTHORIZATIONSESSION_RELEASE_LOG("shouldInterruptLoadForXFrameOptions: %s", errorMessage.utf8().data());
        return true;
    }
    case XFrameOptionsDisposition::Invalid: {
        auto errorMessage = makeString("Invalid 'X-Frame-Options' header encountered: '"_s, xFrameOptions, "' is not a recognized directive. The header will be ignored."_s);
        AUTHORIZATIONSESSION_RELEASE_LOG("shouldInterruptLoadForXFrameOptions: %s", errorMessage.utf8().data());
        return false;
    }
    }
    ASSERT_NOT_REACHED();
    return false;
}

bool SubFrameSOAuthorizationSession::shouldInterruptLoadForCSPFrameAncestorsOrXFrameOptions(const WebCore::ResourceResponse& response)
{
    if (RefPtr page = this->page(); page && page->protectedPreferences()->ignoreIframeEmbeddingProtectionsEnabled())
        return false;

    Vector<Ref<SecurityOrigin>> frameAncestorOrigins;

    ASSERT(navigationAction());
    if (auto* targetFrame = navigationAction()->targetFrame()) {
        if (auto parentFrameHandle = targetFrame->parentFrameHandle()) {
            for (auto* parent = WebFrameProxy::webFrame(parentFrameHandle->frameID()); parent; parent = parent->parentFrame())
                frameAncestorOrigins.append(SecurityOrigin::create(parent->url()));
        }
    }

    auto url = response.url();
    ContentSecurityPolicy contentSecurityPolicy { URL { url }, nullptr, nullptr };
    contentSecurityPolicy.didReceiveHeaders(ContentSecurityPolicyResponseHeaders { response }, navigationAction()->request().httpReferrer());
    if (!contentSecurityPolicy.allowFrameAncestors(frameAncestorOrigins, url))
        return true;

    if (!contentSecurityPolicy.overridesXFrameOptions()) {
        String xFrameOptions = response.httpHeaderField(HTTPHeaderName::XFrameOptions);
        if (!xFrameOptions.isNull() && shouldInterruptLoadForXFrameOptions(WTFMove(frameAncestorOrigins), xFrameOptions, response.url())) {
            String errorMessage = makeString("Refused to display '"_s, response.url().stringCenterEllipsizedToLength(), "' in a frame because it set 'X-Frame-Options' to '"_s, xFrameOptions, "'."_s);
            AUTHORIZATIONSESSION_RELEASE_LOG("shouldInterruptLoadForCSPFrameAncestorsOrXFrameOptions: %s", errorMessage.utf8().data());

            return true;
        }
    }

    return false;
}

} // namespace WebKit

#undef AUTHORIZATIONSESSION_RELEASE_LOG

#endif
