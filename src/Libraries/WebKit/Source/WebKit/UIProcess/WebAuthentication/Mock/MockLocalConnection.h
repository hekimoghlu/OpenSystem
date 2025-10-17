/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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

#include "LocalConnection.h"
#include <WebCore/AuthenticatorAssertionResponse.h>
#include <WebCore/MockWebAuthenticationConfiguration.h>

namespace WebKit {

class MockLocalConnection final : public LocalConnection {
    WTF_MAKE_TZONE_ALLOCATED(MockLocalConnection);
public:
    static Ref<MockLocalConnection> create(const WebCore::MockWebAuthenticationConfiguration&);

private:
    explicit MockLocalConnection(const WebCore::MockWebAuthenticationConfiguration&);

    RetainPtr<NSArray> getExistingCredentials(const String& rpId) final;
    void verifyUser(const String&, WebCore::ClientDataType, SecAccessControlRef, WebCore::UserVerificationRequirement,  UserVerificationCallback&&) final;
    void verifyUser(SecAccessControlRef, LAContext *, CompletionHandler<void(UserVerification)>&&) final;
    RetainPtr<SecKeyRef> createCredentialPrivateKey(LAContext *, SecAccessControlRef, const String& secAttrLabel, NSData *secAttrApplicationTag) const final;
    void filterResponses(Vector<Ref<WebCore::AuthenticatorAssertionResponse>>&) const final;

    WebCore::MockWebAuthenticationConfiguration m_configuration;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
