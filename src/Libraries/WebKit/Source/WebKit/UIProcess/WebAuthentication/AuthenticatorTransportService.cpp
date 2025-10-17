/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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
#include "AuthenticatorTransportService.h"

#if ENABLE(WEB_AUTHN)

#include "CcidService.h"
#include "HidService.h"
#include "LocalService.h"
#include "MockCcidService.h"
#include "MockHidService.h"
#include "MockLocalService.h"
#include "MockNfcService.h"
#include "NfcService.h"
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AuthenticatorTransportService);

Ref<AuthenticatorTransportService> AuthenticatorTransportService::create(WebCore::AuthenticatorTransport transport, AuthenticatorTransportServiceObserver& observer)
{
    switch (transport) {
    case WebCore::AuthenticatorTransport::Internal:
        return LocalService::create(observer);
    case WebCore::AuthenticatorTransport::Usb:
        return HidService::create(observer);
    case WebCore::AuthenticatorTransport::Nfc:
        return NfcService::create(observer);
    case WebCore::AuthenticatorTransport::SmartCard:
        return CcidService::create(observer);
    default:
        ASSERT_NOT_REACHED();
        return LocalService::create(observer);
    }
}

Ref<AuthenticatorTransportService> AuthenticatorTransportService::createMock(WebCore::AuthenticatorTransport transport, AuthenticatorTransportServiceObserver& observer, const WebCore::MockWebAuthenticationConfiguration& configuration)
{
    switch (transport) {
    case WebCore::AuthenticatorTransport::Internal:
        return MockLocalService::create(observer, configuration);
    case WebCore::AuthenticatorTransport::Usb:
        return MockHidService::create(observer, configuration);
    case WebCore::AuthenticatorTransport::Nfc:
        return MockNfcService::create(observer, configuration);
    case WebCore::AuthenticatorTransport::SmartCard:
        return MockCcidService::create(observer, configuration);
    default:
        ASSERT_NOT_REACHED();
        return MockLocalService::create(observer, configuration);
    }
}

AuthenticatorTransportService::AuthenticatorTransportService(AuthenticatorTransportServiceObserver& observer)
    : m_observer(observer)
{
}

void AuthenticatorTransportService::startDiscovery()
{
    RunLoop::main().dispatch([weakThis = WeakPtr { *this }] {
        if (!weakThis)
            return;
        weakThis->startDiscoveryInternal();
    });
}

void AuthenticatorTransportService::restartDiscovery()
{
    RunLoop::main().dispatch([weakThis = WeakPtr { *this }] {
        if (!weakThis)
            return;
        weakThis->restartDiscoveryInternal();
    });
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
