/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 27, 2024.
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

#include "IDLTypes.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebAuthn {
enum class Scope;
}

namespace WebCore {
class AuthenticatorCoordinator;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::AuthenticatorCoordinator> : std::true_type { };
}

namespace WebCore {

class AbortSignal;
class AuthenticatorCoordinatorClient;
class BasicCredential;
class Document;
typedef IDLRecord<IDLDOMString, IDLBoolean> PublicKeyCredentialClientCapabilities;

struct PublicKeyCredentialCreationOptions;
struct PublicKeyCredentialRequestOptions;
struct CredentialCreationOptions;
struct CredentialRequestOptions;
class SecurityOriginData;

template<typename IDLType> class DOMPromiseDeferred;

using CredentialPromise = DOMPromiseDeferred<IDLNullable<IDLInterface<BasicCredential>>>;
using ScopeAndCrossOriginParent = std::pair<WebAuthn::Scope, std::optional<SecurityOriginData>>;

class AuthenticatorCoordinator final : public CanMakeWeakPtr<AuthenticatorCoordinator> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(AuthenticatorCoordinator, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(AuthenticatorCoordinator);
public:
    WEBCORE_EXPORT explicit AuthenticatorCoordinator(std::unique_ptr<AuthenticatorCoordinatorClient>&&);
    WEBCORE_EXPORT void setClient(std::unique_ptr<AuthenticatorCoordinatorClient>&&);

    // The following methods implement static methods of PublicKeyCredential.
    void create(const Document&, CredentialCreationOptions&&, RefPtr<AbortSignal>&&, CredentialPromise&&);
    void discoverFromExternalSource(const Document&, CredentialRequestOptions&&, CredentialPromise&&);
    void isUserVerifyingPlatformAuthenticatorAvailable(const Document&, DOMPromiseDeferred<IDLBoolean>&&) const;
    void isConditionalMediationAvailable(const Document&, DOMPromiseDeferred<IDLBoolean>&&) const;

    void getClientCapabilities(const Document&, DOMPromiseDeferred<PublicKeyCredentialClientCapabilities>&&) const;

private:
    AuthenticatorCoordinator() = default;

    std::unique_ptr<AuthenticatorCoordinatorClient> m_client;
    bool m_isCancelling = false;
    CompletionHandler<void()> m_queuedRequest;
};

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
