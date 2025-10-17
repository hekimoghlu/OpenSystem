/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
#import "config.h"
#import "VirtualLocalConnection.h"

#if ENABLE(WEB_AUTHN)

#import "VirtualAuthenticatorConfiguration.h"
#import <JavaScriptCore/ArrayBuffer.h>
#import <Security/SecItem.h>
#import <WebCore/AuthenticatorAssertionResponse.h>
#import <WebCore/ExceptionData.h>
#import <wtf/Ref.h>
#import <wtf/RunLoop.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/spi/cocoa/SecuritySPI.h>
#import <wtf/text/Base64.h>
#import <wtf/text/WTFString.h>

#import "LocalAuthenticationSoftLink.h"

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(VirtualLocalConnection);

Ref<VirtualLocalConnection> VirtualLocalConnection::create(const VirtualAuthenticatorConfiguration& configuration)
{
    return adoptRef(*new VirtualLocalConnection(configuration));
}

VirtualLocalConnection::VirtualLocalConnection(const VirtualAuthenticatorConfiguration& configuration)
    : m_configuration(configuration)
{
}

void VirtualLocalConnection::verifyUser(const String&, ClientDataType, SecAccessControlRef, WebCore::UserVerificationRequirement, UserVerificationCallback&& callback)
{
    // Mock async operations.
    RunLoop::main().dispatch([weakThis = WeakPtr { *this }, callback = WTFMove(callback)]() mutable {
        RefPtr protectedThis = weakThis.get();
        if (!protectedThis) {
            callback(UserVerification::No, adoptNS([allocLAContextInstance() init]).get());
            return;
        }
        ASSERT(protectedThis->m_configuration.transport == AuthenticatorTransport::Internal);

        UserVerification userVerification = protectedThis->m_configuration.isUserVerified ? UserVerification::Yes : UserVerification::Presence;

        callback(userVerification, adoptNS([allocLAContextInstance() init]).get());
    });
}

void VirtualLocalConnection::verifyUser(SecAccessControlRef, LAContext *, CompletionHandler<void(UserVerification)>&& callback)
{
    // Mock async operations.
    RunLoop::main().dispatch([weakThis = WeakPtr { *this }, callback = WTFMove(callback)]() mutable {
        RefPtr protectedThis = weakThis.get();
        if (!protectedThis) {
            callback(UserVerification::No);
            return;
        }
        ASSERT(protectedThis->m_configuration.transport == AuthenticatorTransport::Internal);

        UserVerification userVerification = protectedThis->m_configuration.isUserVerified ? UserVerification::Yes : UserVerification::Presence;

        callback(userVerification);
    });
}

void VirtualLocalConnection::filterResponses(Vector<Ref<AuthenticatorAssertionResponse>>& responses) const
{
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
