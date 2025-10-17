/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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

#if ENABLE(WEB_AUTHN)

#include <WebCore/AuthenticatorCoordinatorClient.h>
#include <WebCore/FrameIdentifier.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebKit {

class WebPage;

class WebAuthenticatorCoordinator final : public WebCore::AuthenticatorCoordinatorClient {
    WTF_MAKE_TZONE_ALLOCATED(WebAuthenticatorCoordinator);
public:
    explicit WebAuthenticatorCoordinator(WebPage&);

private:
    // WebCore::AuthenticatorCoordinatorClient
    void makeCredential(const WebCore::LocalFrame&, const WebCore::PublicKeyCredentialCreationOptions&, WebCore::MediationRequirement, WebCore::RequestCompletionHandler&&) final;
    void getAssertion(const WebCore::LocalFrame&, const WebCore::PublicKeyCredentialRequestOptions&, WebCore::MediationRequirement, const std::pair<WebAuthn::Scope, std::optional<WebCore::SecurityOriginData>>&, WebCore::RequestCompletionHandler&&) final;
    void isConditionalMediationAvailable(const WebCore::SecurityOrigin&, WebCore::QueryCompletionHandler&&) final;
    void isUserVerifyingPlatformAuthenticatorAvailable(const WebCore::SecurityOrigin&, WebCore::QueryCompletionHandler&&) final;
    void getClientCapabilities(const WebCore::SecurityOrigin&, WebCore::CapabilitiesCompletionHandler&&) final;
    void cancel(CompletionHandler<void()>&&) final;

    Ref<WebPage> protectedPage() const;

    WeakRef<WebPage> m_webPage;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
