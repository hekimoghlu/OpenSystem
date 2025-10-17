/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
#import "LocalService.h"

#if ENABLE(WEB_AUTHN)

#import "LocalAuthenticator.h"
#import "LocalConnection.h"
#import <wtf/TZoneMallocInlines.h>

#if USE(APPLE_INTERNAL_SDK)
#import <WebKitAdditions/LocalServiceAdditions.h>
#else
#define LOCAL_SERVICE_ADDITIONS
#endif

#import "LocalAuthenticationSoftLink.h"

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LocalService);

Ref<LocalService> LocalService::create(AuthenticatorTransportServiceObserver& observer)
{
    return adoptRef(*new LocalService(observer));
}

LocalService::LocalService(AuthenticatorTransportServiceObserver& observer)
    : AuthenticatorTransportService(observer)
{
}

bool LocalService::isAvailable()
{
LOCAL_SERVICE_ADDITIONS

    auto context = adoptNS([allocLAContextInstance() init]);
    NSError *error = nil;
    auto result = [context canEvaluatePolicy:LAPolicyDeviceOwnerAuthenticationWithBiometrics error:&error];
    if ((!result || error) && error.code != LAErrorBiometryLockout) {
        LOG_ERROR("Couldn't find local authenticators: %@", error);
        return false;
    }

    return true;
}

void LocalService::startDiscoveryInternal()
{
    if (!platformStartDiscovery() || !observer())
        return;
    observer()->authenticatorAdded(LocalAuthenticator::create(createLocalConnection()));
}

bool LocalService::platformStartDiscovery() const
{
    return LocalService::isAvailable();
}

Ref<LocalConnection> LocalService::createLocalConnection() const
{
    return LocalConnection::create();
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
