/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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
#import "MockLocalService.h"

#if ENABLE(WEB_AUTHN)

#import "LocalAuthenticator.h"
#import "MockLocalConnection.h"
#import <wtf/RunLoop.h>

#if USE(APPLE_INTERNAL_SDK)
#import <WebKitAdditions/MockLocalServiceAdditions.h>
#else
#define MOCK_LOCAL_SERVICE_ADDITIONS
#endif

namespace WebKit {

Ref<MockLocalService> MockLocalService::create(AuthenticatorTransportServiceObserver& observer, const WebCore::MockWebAuthenticationConfiguration& configuration)
{
    return adoptRef(*new MockLocalService(observer, configuration));
}

MockLocalService::MockLocalService(AuthenticatorTransportServiceObserver& observer, const WebCore::MockWebAuthenticationConfiguration& configuration)
    : LocalService(observer)
    , m_configuration(configuration)
{
MOCK_LOCAL_SERVICE_ADDITIONS
}

bool MockLocalService::platformStartDiscovery() const
{
    return !!m_configuration.local;
}

Ref<LocalConnection> MockLocalService::createLocalConnection() const
{
    return MockLocalConnection::create(m_configuration);
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
