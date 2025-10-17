/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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
#import "VirtualService.h"

#if ENABLE(WEB_AUTHN)

#import "CtapAuthenticator.h"
#import "CtapHidDriver.h"
#import "LocalAuthenticator.h"
#import "VirtualAuthenticatorManager.h"
#import "VirtualHidConnection.h"
#import "VirtualLocalConnection.h"
#import <WebCore/FidoConstants.h>
#import <WebCore/WebAuthenticationConstants.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/UniqueRef.h>
#import <wtf/text/WTFString.h>

namespace WebKit {
using namespace fido;
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(VirtualService);

Ref<VirtualService> VirtualService::create(AuthenticatorTransportServiceObserver& observer, Vector<std::pair<String, VirtualAuthenticatorConfiguration>>& configuration)
{
    return adoptRef(*new VirtualService(observer, configuration));
}

VirtualService::VirtualService(AuthenticatorTransportServiceObserver& observer, Vector<std::pair<String, VirtualAuthenticatorConfiguration>>& authenticators)
    : AuthenticatorTransportService(observer), m_authenticators(authenticators)
{
}

Ref<AuthenticatorTransportService> VirtualService::createVirtual(WebCore::AuthenticatorTransport transport, AuthenticatorTransportServiceObserver& observer, Vector<std::pair<String, VirtualAuthenticatorConfiguration>>& authenticators)
{
    return VirtualService::create(observer, authenticators);
}

static AuthenticatorGetInfoResponse authenticatorInfoForConfig(const VirtualAuthenticatorConfiguration& config)
{
    AuthenticatorGetInfoResponse infoResponse({ ProtocolVersion::kCtap2 }, Vector<uint8_t>(aaguidLength, 0u));
    AuthenticatorSupportedOptions options;
    infoResponse.setOptions(WTFMove(options));
    return infoResponse;
}

void VirtualService::startDiscoveryInternal()
{

    for (auto& authenticator : m_authenticators) {
        if (!observer())
            return;
        auto config = authenticator.second;
        auto authenticatorId = authenticator.first;
        switch (config.transport) {
        case WebCore::AuthenticatorTransport::Nfc:
        case WebCore::AuthenticatorTransport::Ble:
        case WebCore::AuthenticatorTransport::Usb:
            observer()->authenticatorAdded(CtapAuthenticator::create(CtapHidDriver::create(VirtualHidConnection::create(authenticatorId, config, WeakPtr { static_cast<VirtualAuthenticatorManager *>(observer()) })), authenticatorInfoForConfig(config)));
            break;
        case WebCore::AuthenticatorTransport::Internal:
            observer()->authenticatorAdded(LocalAuthenticator::create(VirtualLocalConnection::create(config)));
            break;
        default:
            UNIMPLEMENTED();
        }
    }
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
