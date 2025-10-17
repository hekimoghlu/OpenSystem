/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
#include "IDLTypes.h"
#include <wtf/RefCounted.h>
#include <wtf/TypeCasts.h>

namespace WebCore {

enum class AuthenticatorAttachment : uint8_t;
enum class AuthenticatorTransport : uint8_t;

struct AuthenticatorResponseData;

class AuthenticatorResponse : public RefCounted<AuthenticatorResponse> {
public:
    enum class Type {
        Assertion,
        Attestation
    };

    static RefPtr<AuthenticatorResponse> tryCreate(AuthenticatorResponseData&&, AuthenticatorAttachment);
    virtual ~AuthenticatorResponse() = default;

    virtual Type type() const = 0;
    virtual AuthenticatorResponseData data() const;

    WEBCORE_EXPORT ArrayBuffer* rawId() const;
    WEBCORE_EXPORT void setExtensions(AuthenticationExtensionsClientOutputs&&);
    WEBCORE_EXPORT AuthenticationExtensionsClientOutputs extensions() const;
    WEBCORE_EXPORT void setClientDataJSON(Ref<ArrayBuffer>&&);
    ArrayBuffer* clientDataJSON() const;
    WEBCORE_EXPORT AuthenticatorAttachment attachment() const;

protected:
    AuthenticatorResponse(Ref<ArrayBuffer>&&, AuthenticatorAttachment);

private:
    Ref<ArrayBuffer> m_rawId;
    AuthenticationExtensionsClientOutputs m_extensions;
    RefPtr<ArrayBuffer> m_clientDataJSON;
    AuthenticatorAttachment m_attachment;
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_AUTHENTICATOR_RESPONSE(ToClassName, Type) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToClassName) \
    static bool isType(const WebCore::AuthenticatorResponse& response) { return response.type() == WebCore::Type; } \
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(WEB_AUTHN)
