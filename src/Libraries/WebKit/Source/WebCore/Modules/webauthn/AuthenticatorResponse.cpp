/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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
#include "AuthenticatorResponse.h"

#if ENABLE(WEB_AUTHN)

#include "AuthenticatorAssertionResponse.h"
#include "AuthenticatorAttestationResponse.h"
#include "AuthenticatorResponseData.h"

namespace WebCore {

RefPtr<AuthenticatorResponse> AuthenticatorResponse::tryCreate(AuthenticatorResponseData&& data, AuthenticatorAttachment attachment)
{
    if (!data.rawId)
        return nullptr;

    if (data.isAuthenticatorAttestationResponse) {
        if (!data.attestationObject)
            return nullptr;

        auto response = AuthenticatorAttestationResponse::create(data.rawId.releaseNonNull(), data.attestationObject.releaseNonNull(), attachment, WTFMove(data.transports));
        if (data.extensionOutputs)
            response->setExtensions(WTFMove(*data.extensionOutputs));
        response->setClientDataJSON(data.clientDataJSON.releaseNonNull());
        return WTFMove(response);
    }

    if (!data.authenticatorData || !data.signature)
        return nullptr;

    Ref response = AuthenticatorAssertionResponse::create(data.rawId.releaseNonNull(), data.authenticatorData.releaseNonNull(), data.signature.releaseNonNull(), WTFMove(data.userHandle), WTFMove(data.extensionOutputs), attachment);
    response->setClientDataJSON(data.clientDataJSON.releaseNonNull());
    return WTFMove(response);
}

AuthenticatorResponseData AuthenticatorResponse::data() const
{
    AuthenticatorResponseData data;
    data.rawId = m_rawId.copyRef();
    data.extensionOutputs = m_extensions;
    data.clientDataJSON = m_clientDataJSON.copyRef();
    return data;
}

ArrayBuffer* AuthenticatorResponse::rawId() const
{
    return m_rawId.ptr();
}

void AuthenticatorResponse::setExtensions(AuthenticationExtensionsClientOutputs&& extensions)
{
    m_extensions = WTFMove(extensions);
}

AuthenticationExtensionsClientOutputs AuthenticatorResponse::extensions() const
{
    return m_extensions;
}

void AuthenticatorResponse::setClientDataJSON(Ref<ArrayBuffer>&& clientDataJSON)
{
    m_clientDataJSON = WTFMove(clientDataJSON);
}

ArrayBuffer* AuthenticatorResponse::clientDataJSON() const
{
    return m_clientDataJSON.get();
}

AuthenticatorAttachment AuthenticatorResponse::attachment() const
{
    return m_attachment;
}

AuthenticatorResponse::AuthenticatorResponse(Ref<ArrayBuffer>&& rawId, AuthenticatorAttachment attachment)
    : m_rawId(WTFMove(rawId))
    , m_attachment(attachment)
{
}

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
