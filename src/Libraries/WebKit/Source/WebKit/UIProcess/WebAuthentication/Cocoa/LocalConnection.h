/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 20, 2024.
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

#include <wtf/CompletionHandler.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS LAContext;

namespace WebCore {
class AuthenticatorAssertionResponse;
enum class ClientDataType : bool;
enum class UserVerificationRequirement : uint8_t;
}

namespace WebKit {

// Local authenticators normally doesn't need to establish connections
// between the platform and themselves as they are attached.
// However, such abstraction is still provided to isolate operations
// that are not allowed in auto test environment such that some mocking
// mechnism can override them.
class LocalConnection : public RefCounted<LocalConnection> {
    WTF_MAKE_TZONE_ALLOCATED(LocalConnection);
    WTF_MAKE_NONCOPYABLE(LocalConnection);
public:
    static Ref<LocalConnection> create();

    enum class UserVerification : uint8_t {
        No,
        Yes,
        Cancel,
        Presence
    };

    using AttestationCallback = CompletionHandler<void(NSArray *, NSError *)>;
    using UserVerificationCallback = CompletionHandler<void(UserVerification, LAContext *)>;

    virtual ~LocalConnection();

    // Overrided by MockLocalConnection.
    virtual RetainPtr<NSArray> getExistingCredentials(const String& rpId);
    virtual void verifyUser(const String& rpId, WebCore::ClientDataType, SecAccessControlRef, WebCore::UserVerificationRequirement, UserVerificationCallback&&);
    virtual void verifyUser(SecAccessControlRef, LAContext *, CompletionHandler<void(UserVerification)>&&);
    virtual RetainPtr<SecKeyRef> createCredentialPrivateKey(LAContext *, SecAccessControlRef, const String& secAttrLabel, NSData *secAttrApplicationTag) const;
    virtual void filterResponses(Vector<Ref<WebCore::AuthenticatorAssertionResponse>>&) const { };

protected:
    LocalConnection() = default;

private:
    RetainPtr<LAContext> m_context;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
