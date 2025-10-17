/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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
#include "MockHidService.h"

#if ENABLE(WEB_AUTHN)

#include "MockHidConnection.h"
#include <wtf/RunLoop.h>

namespace WebKit {

Ref<MockHidService> MockHidService::create(AuthenticatorTransportServiceObserver& observer, const WebCore::MockWebAuthenticationConfiguration& configuration)
{
    return adoptRef(*new MockHidService(observer, configuration));
}

MockHidService::MockHidService(AuthenticatorTransportServiceObserver& observer, const WebCore::MockWebAuthenticationConfiguration& configuration)
    : HidService(observer)
    , m_configuration(configuration)
{
}

void MockHidService::platformStartDiscovery()
{
    if (!!m_configuration.hid) {
        deviceAdded(nullptr);
        return;
    }
    LOG_ERROR("No hid authenticators is available.");
}

Ref<HidConnection> MockHidService::createHidConnection(IOHIDDeviceRef device) const
{
    return MockHidConnection::create(device, m_configuration);
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
