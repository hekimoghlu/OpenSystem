/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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
#include "config.h"
#include "WebAuthenticatorCoordinator.h"

#if ENABLE(WEB_AUTHN)

#include "DefaultWebBrowserChecks.h"
#include "FrameInfoData.h"
#include "WebAuthenticatorCoordinatorProxyMessages.h"
#include "WebFrame.h"
#include "WebPage.h"
#include "WebProcess.h"
#include <JavaScriptCore/ConsoleTypes.h>
#include <WebCore/AuthenticatorAttachment.h>
#include <WebCore/AuthenticatorResponseData.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/MediationRequirement.h>
#include <WebCore/PublicKeyCredentialCreationOptions.h>
#include <WebCore/PublicKeyCredentialRequestOptions.h>
#include <WebCore/Quirks.h>
#include <WebCore/SecurityOrigin.h>
#include <WebCore/UserGestureIndicator.h>
#include <WebCore/WebAuthenticationConstants.h>
#include <wtf/TZoneMallocInlines.h>

#undef WEBAUTHN_RELEASE_LOG
#define PAGE_ID (m_webPage->identifier().toUInt64())
#define FRAME_ID (webFrame->frameID().object().toUInt64())
#define WEBAUTHN_RELEASE_LOG_ERROR(fmt, ...) RELEASE_LOG_ERROR(WebAuthn, "%p - [webPageID=%" PRIu64 ", webFrameID=%" PRIu64 "] WebAuthenticatorCoordinator::" fmt, this, PAGE_ID, FRAME_ID, ##__VA_ARGS__)
#define WEBAUTHN_RELEASE_LOG_ERROR_NO_FRAME(fmt, ...) RELEASE_LOG_ERROR(WebAuthn, "%p - [webPageID=%" PRIu64 "] WebAuthenticatorCoordinator::" fmt, this, PAGE_ID, ##__VA_ARGS__)

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebAuthenticatorCoordinator);

namespace {
inline bool isWebBrowser()
{
    return isParentProcessAFullWebBrowser(WebProcess::singleton());
}
}

WebAuthenticatorCoordinator::WebAuthenticatorCoordinator(WebPage& webPage)
    : m_webPage(webPage)
{
}

Ref<WebPage> WebAuthenticatorCoordinator::protectedPage() const
{
    return m_webPage.get();
}

void WebAuthenticatorCoordinator::makeCredential(const LocalFrame& frame, const PublicKeyCredentialCreationOptions& options, MediationRequirement mediation, RequestCompletionHandler&& handler)
{
    auto webFrame = WebFrame::fromCoreFrame(frame);
    if (!webFrame)
        return;

    protectedPage()->sendWithAsyncReply(Messages::WebAuthenticatorCoordinatorProxy::MakeCredential(webFrame->frameID(), webFrame->info(), options, mediation), WTFMove(handler));
}

void WebAuthenticatorCoordinator::getAssertion(const LocalFrame& frame, const PublicKeyCredentialRequestOptions& options, MediationRequirement mediation, const ScopeAndCrossOriginParent& scopeAndCrossOriginParent, RequestCompletionHandler&& handler)
{
    auto webFrame = WebFrame::fromCoreFrame(frame);
    if (!webFrame)
        return;

    protectedPage()->sendWithAsyncReply(Messages::WebAuthenticatorCoordinatorProxy::GetAssertion(webFrame->frameID(), webFrame->info(), options, mediation, scopeAndCrossOriginParent.second), WTFMove(handler));
}

void WebAuthenticatorCoordinator::isConditionalMediationAvailable(const SecurityOrigin& origin, QueryCompletionHandler&& handler)
{
    protectedPage()->sendWithAsyncReply(Messages::WebAuthenticatorCoordinatorProxy::isConditionalMediationAvailable(origin.data()), WTFMove(handler));
};

void WebAuthenticatorCoordinator::isUserVerifyingPlatformAuthenticatorAvailable(const SecurityOrigin& origin, QueryCompletionHandler&& handler)
{
    protectedPage()->sendWithAsyncReply(Messages::WebAuthenticatorCoordinatorProxy::IsUserVerifyingPlatformAuthenticatorAvailable(origin.data()), WTFMove(handler));
}

void WebAuthenticatorCoordinator::cancel(CompletionHandler<void()>&& handler)
{
    protectedPage()->sendWithAsyncReply(Messages::WebAuthenticatorCoordinatorProxy::Cancel(), WTFMove(handler));
}

void WebAuthenticatorCoordinator::getClientCapabilities(const SecurityOrigin& origin, CapabilitiesCompletionHandler&& handler)
{
    protectedPage()->sendWithAsyncReply(Messages::WebAuthenticatorCoordinatorProxy::GetClientCapabilities(origin.data()), WTFMove(handler));
}

} // namespace WebKit

#undef WEBAUTHN_RELEASE_LOG_ERROR_NO_FRAME
#undef WEBAUTHN_RELEASE_LOG_ERROR
#undef FRAME_ID
#undef PAGE_ID

#endif // ENABLE(WEB_AUTHN)
