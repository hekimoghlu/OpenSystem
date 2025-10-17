/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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

#include "AuthenticatorCoordinator.h"
#include "ExceptionData.h"
#include <wtf/CompletionHandler.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class AuthenticatorCoordinatorClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::AuthenticatorCoordinatorClient> : std::true_type { };
}

namespace WebAuthn {
enum class Scope;
}

namespace WebCore {

class DeferredPromise;
class LocalFrame;
class SecurityOrigin;

enum class AuthenticatorAttachment : uint8_t;
enum class MediationRequirement : uint8_t;

struct AuthenticatorResponseData;
struct PublicKeyCredentialCreationOptions;
struct PublicKeyCredentialRequestOptions;
class SecurityOriginData;

using CapabilitiesCompletionHandler = CompletionHandler<void(Vector<KeyValuePair<String, bool>>&&)>;
using RequestCompletionHandler = CompletionHandler<void(WebCore::AuthenticatorResponseData&&, WebCore::AuthenticatorAttachment, WebCore::ExceptionData&&)>;
using QueryCompletionHandler = CompletionHandler<void(bool)>;

class AuthenticatorCoordinatorClient : public CanMakeWeakPtr<AuthenticatorCoordinatorClient> {
    WTF_MAKE_TZONE_ALLOCATED(AuthenticatorCoordinatorClient);
    WTF_MAKE_NONCOPYABLE(AuthenticatorCoordinatorClient);
public:
    AuthenticatorCoordinatorClient() = default;
    virtual ~AuthenticatorCoordinatorClient() = default;

    virtual void makeCredential(const LocalFrame&, const PublicKeyCredentialCreationOptions&, MediationRequirement, RequestCompletionHandler&&) = 0;
    virtual void getAssertion(const LocalFrame&, const PublicKeyCredentialRequestOptions&, MediationRequirement, const ScopeAndCrossOriginParent&, RequestCompletionHandler&&) = 0;
    virtual void isConditionalMediationAvailable(const SecurityOrigin&, QueryCompletionHandler&&) = 0;
    virtual void isUserVerifyingPlatformAuthenticatorAvailable(const SecurityOrigin&, QueryCompletionHandler&&) = 0;
    virtual void getClientCapabilities(const SecurityOrigin&, CapabilitiesCompletionHandler&&) = 0;
    virtual void cancel(CompletionHandler<void()>&&) = 0;
};

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
