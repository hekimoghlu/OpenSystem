/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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

#include "AuthenticationExtensionsClientOutputs.h"
#include "AuthenticatorTransport.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <wtf/Forward.h>

namespace WebCore {

class AuthenticatorResponse;

struct AuthenticatorResponseBaseData {
    RefPtr<ArrayBuffer> rawId;
    std::optional<AuthenticationExtensionsClientOutputs> extensionOutputs;
};

struct AuthenticatorAttestationResponseData {
    RefPtr<ArrayBuffer> rawId;
    std::optional<AuthenticationExtensionsClientOutputs> extensionOutputs;
    RefPtr<ArrayBuffer> clientDataJSON;
    RefPtr<ArrayBuffer> attestationObject;
    Vector<WebCore::AuthenticatorTransport> transports;
};

struct AuthenticatorAssertionResponseData {
    RefPtr<ArrayBuffer> rawId;
    std::optional<AuthenticationExtensionsClientOutputs> extensionOutputs;
    RefPtr<ArrayBuffer> clientDataJSON;
    RefPtr<ArrayBuffer> authenticatorData;
    RefPtr<ArrayBuffer> signature;
    RefPtr<ArrayBuffer> userHandle;
};

using AuthenticatorResponseDataSerializableForm = std::variant<std::nullptr_t, AuthenticatorResponseBaseData, AuthenticatorAttestationResponseData, AuthenticatorAssertionResponseData>;

struct AuthenticatorResponseData {
    AuthenticatorResponseData() = default;
    AuthenticatorResponseData(const AuthenticatorResponseDataSerializableForm& data)
    {
        WTF::switchOn(data, [](std::nullptr_t) {
        }, [&](const AuthenticatorResponseBaseData& v) {
            rawId = v.rawId;
            extensionOutputs = v.extensionOutputs;
        }, [&](const AuthenticatorAttestationResponseData& v) {
            isAuthenticatorAttestationResponse = true;
            rawId = v.rawId;
            extensionOutputs = v.extensionOutputs;
            clientDataJSON = v.clientDataJSON;
            attestationObject = v.attestationObject;
            transports = v.transports;
        }, [&](const AuthenticatorAssertionResponseData& v) {
            rawId = v.rawId;
            extensionOutputs = v.extensionOutputs;
            clientDataJSON = v.clientDataJSON;
            authenticatorData = v.authenticatorData;
            signature = v.signature;
            userHandle = v.userHandle;
        });
    }

    bool isAuthenticatorAttestationResponse { false };

    // AuthenticatorResponse
    RefPtr<ArrayBuffer> rawId;

    // Extensions
    std::optional<AuthenticationExtensionsClientOutputs> extensionOutputs;

    RefPtr<ArrayBuffer> clientDataJSON;

    // AuthenticatorAttestationResponse
    RefPtr<ArrayBuffer> attestationObject;

    // AuthenticatorAssertionResponse
    RefPtr<ArrayBuffer> authenticatorData;
    RefPtr<ArrayBuffer> signature;
    RefPtr<ArrayBuffer> userHandle;

    Vector<WebCore::AuthenticatorTransport> transports;

    AuthenticatorResponseDataSerializableForm getSerializableForm() const
    {
        if (!rawId)
            return nullptr;

        if (isAuthenticatorAttestationResponse && attestationObject)
            return AuthenticatorAttestationResponseData { rawId, extensionOutputs, clientDataJSON, attestationObject, transports };

        if (!authenticatorData || !signature)
            return AuthenticatorResponseBaseData { rawId, extensionOutputs };

        return AuthenticatorAssertionResponseData { rawId, extensionOutputs, clientDataJSON, authenticatorData, signature, userHandle };
    }
};
    
} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
