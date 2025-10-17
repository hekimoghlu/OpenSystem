/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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
#include "AuthenticatorAssertionResponse.h"

#if ENABLE(WEB_AUTHN)

#include "AuthenticatorResponseData.h"
#include <wtf/text/Base64.h>

namespace WebCore {

Ref<AuthenticatorAssertionResponse> AuthenticatorAssertionResponse::create(Ref<ArrayBuffer>&& rawId, Ref<ArrayBuffer>&& authenticatorData, Ref<ArrayBuffer>&& signature, RefPtr<ArrayBuffer>&& userHandle, std::optional<AuthenticationExtensionsClientOutputs>&& extensions, AuthenticatorAttachment attachment)
{
    auto response = adoptRef(*new AuthenticatorAssertionResponse(WTFMove(rawId), WTFMove(authenticatorData), WTFMove(signature), WTFMove(userHandle), attachment));
    if (extensions)
        response->setExtensions(WTFMove(*extensions));
    return response;
}

Ref<AuthenticatorAssertionResponse> AuthenticatorAssertionResponse::create(const Vector<uint8_t>& rawId, const Vector<uint8_t>& authenticatorData, const Vector<uint8_t>& signature, const Vector<uint8_t>& userHandle, AuthenticatorAttachment attachment)
{
    RefPtr<ArrayBuffer> userhandleBuffer;
    if (!userHandle.isEmpty())
        userhandleBuffer = ArrayBuffer::create(userHandle);
    return create(ArrayBuffer::create(rawId), ArrayBuffer::create(authenticatorData), ArrayBuffer::create(signature), WTFMove(userhandleBuffer), std::nullopt, attachment);
}

Ref<AuthenticatorAssertionResponse> AuthenticatorAssertionResponse::create(Ref<ArrayBuffer>&& rawId, RefPtr<ArrayBuffer>&& userHandle, String&& name, SecAccessControlRef accessControl, AuthenticatorAttachment attachment)
{
    return adoptRef(*new AuthenticatorAssertionResponse(WTFMove(rawId), WTFMove(userHandle), WTFMove(name), accessControl, attachment));
}

void AuthenticatorAssertionResponse::setAuthenticatorData(Vector<uint8_t>&& authenticatorData)
{
    m_authenticatorData = ArrayBuffer::create(authenticatorData);
}

AuthenticatorAssertionResponse::AuthenticatorAssertionResponse(Ref<ArrayBuffer>&& rawId, Ref<ArrayBuffer>&& authenticatorData, Ref<ArrayBuffer>&& signature, RefPtr<ArrayBuffer>&& userHandle, AuthenticatorAttachment attachment)
    : AuthenticatorResponse(WTFMove(rawId), attachment)
    , m_authenticatorData(WTFMove(authenticatorData))
    , m_signature(WTFMove(signature))
    , m_userHandle(WTFMove(userHandle))
{
}

AuthenticatorAssertionResponse::AuthenticatorAssertionResponse(Ref<ArrayBuffer>&& rawId, RefPtr<ArrayBuffer>&& userHandle, String&& name, SecAccessControlRef accessControl, AuthenticatorAttachment attachment)
    : AuthenticatorResponse(WTFMove(rawId), attachment)
    , m_userHandle(WTFMove(userHandle))
    , m_name(WTFMove(name))
    , m_accessControl(accessControl)
{
}

AuthenticatorResponseData AuthenticatorAssertionResponse::data() const
{
    auto data = AuthenticatorResponse::data();
    data.isAuthenticatorAttestationResponse = false;
    data.authenticatorData = m_authenticatorData.copyRef();
    data.signature = m_signature.copyRef();
    data.userHandle = m_userHandle;
    return data;
}

AuthenticationResponseJSON::AuthenticatorAssertionResponseJSON AuthenticatorAssertionResponse::toJSON()
{
    AuthenticationResponseJSON::AuthenticatorAssertionResponseJSON value;
    if (auto authData = authenticatorData())
        value.authenticatorData = base64URLEncodeToString(authData->span());
    if (auto sig = signature())
        value.signature = base64URLEncodeToString(sig->span());
    if (auto handle = userHandle())
        value.userHandle = base64URLEncodeToString(handle->span());
    if (auto clientData = clientDataJSON())
        value.clientDataJSON = base64URLEncodeToString(clientData->span());
    return value;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
