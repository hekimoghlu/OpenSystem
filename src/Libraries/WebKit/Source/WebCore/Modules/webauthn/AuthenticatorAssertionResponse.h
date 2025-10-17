/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 5, 2022.
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

#include "AuthenticationResponseJSON.h"
#include "AuthenticatorResponse.h"
#include <wtf/RetainPtr.h>
#include <wtf/spi/cocoa/SecuritySPI.h>

OBJC_CLASS LAContext;

namespace WebCore {

class AuthenticatorAssertionResponse : public AuthenticatorResponse {
public:
    static Ref<AuthenticatorAssertionResponse> create(Ref<ArrayBuffer>&& rawId, Ref<ArrayBuffer>&& authenticatorData, Ref<ArrayBuffer>&& signature, RefPtr<ArrayBuffer>&& userHandle, std::optional<AuthenticationExtensionsClientOutputs>&&, AuthenticatorAttachment);
    WEBCORE_EXPORT static Ref<AuthenticatorAssertionResponse> create(const Vector<uint8_t>& rawId, const Vector<uint8_t>& authenticatorData, const Vector<uint8_t>& signature,  const Vector<uint8_t>& userHandle, AuthenticatorAttachment);
    WEBCORE_EXPORT static Ref<AuthenticatorAssertionResponse> create(Ref<ArrayBuffer>&& rawId, RefPtr<ArrayBuffer>&& userHandle, String&& name, SecAccessControlRef, AuthenticatorAttachment);
    virtual ~AuthenticatorAssertionResponse() = default;

    ArrayBuffer* authenticatorData() const { return m_authenticatorData.get(); }
    ArrayBuffer* signature() const { return m_signature.get(); }
    ArrayBuffer* userHandle() const { return m_userHandle.get(); }
    const String& name() const { return m_name; }
    const String& displayName() const { return m_displayName; }
    size_t numberOfCredentials() const { return m_numberOfCredentials; }
    SecAccessControlRef accessControl() const { return m_accessControl.get(); }
    const String& group() const { return m_group; }
    bool synchronizable() const { return m_synchronizable; }
    LAContext * laContext() const { return m_laContext.get(); }
    RefPtr<ArrayBuffer> largeBlob() const { return m_largeBlob; }
    const String& accessGroup() const { return m_accessGroup; }

    WEBCORE_EXPORT void setAuthenticatorData(Vector<uint8_t>&&);
    void setSignature(Ref<ArrayBuffer>&& signature) { m_signature = WTFMove(signature); }
    void setName(const String& name) { m_name = name; }
    void setDisplayName(const String& displayName) { m_displayName = displayName; }
    void setNumberOfCredentials(size_t numberOfCredentials) { m_numberOfCredentials = numberOfCredentials; }
    void setGroup(const String& group) { m_group = group; }
    void setSynchronizable(bool synchronizable) { m_synchronizable = synchronizable; }
    void setLAContext(LAContext *context) { m_laContext = context; }
    void setLargeBlob(Ref<ArrayBuffer>&& largeBlob) { m_largeBlob = WTFMove(largeBlob); }
    void setAccessGroup(const String& accessGroup) { m_accessGroup = accessGroup; }

    AuthenticationResponseJSON::AuthenticatorAssertionResponseJSON toJSON();

private:
    AuthenticatorAssertionResponse(Ref<ArrayBuffer>&&, Ref<ArrayBuffer>&&, Ref<ArrayBuffer>&&, RefPtr<ArrayBuffer>&&, AuthenticatorAttachment);
    AuthenticatorAssertionResponse(Ref<ArrayBuffer>&&, RefPtr<ArrayBuffer>&&, String&&, SecAccessControlRef, AuthenticatorAttachment);

    Type type() const final { return Type::Assertion; }
    AuthenticatorResponseData data() const final;

    RefPtr<ArrayBuffer> m_authenticatorData;
    RefPtr<ArrayBuffer> m_signature;
    RefPtr<ArrayBuffer> m_userHandle;

    String m_name;
    String m_displayName;
    String m_group;
    bool m_synchronizable;
    size_t m_numberOfCredentials { 0 };
    RetainPtr<SecAccessControlRef> m_accessControl;
    RetainPtr<LAContext> m_laContext;
    RefPtr<ArrayBuffer> m_largeBlob;
    String m_accessGroup;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_AUTHENTICATOR_RESPONSE(AuthenticatorAssertionResponse, AuthenticatorResponse::Type::Assertion)

#endif // ENABLE(WEB_AUTHN)
