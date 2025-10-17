/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
#import "NetworkProcess.h"

#if PLATFORM(IOS_FAMILY)

#import "NetworkCache.h"
#import "NetworkProcessCreationParameters.h"
#import "SandboxInitializationParameters.h"
#import "SecItemShim.h"
#import <UIKit/UIKit.h>
#import <WebCore/CertificateInfo.h>
#import <WebCore/NotImplemented.h>
#import <WebCore/WebCoreThreadSystemInterface.h>
#import <pal/spi/cf/CFNetworkSPI.h>
#import <wtf/cocoa/Entitlements.h>

namespace WebKit {

#if !PLATFORM(MACCATALYST)

void NetworkProcess::initializeProcess(const AuxiliaryProcessInitializationParameters&)
{
    InitWebCoreThreadSystemInterface();
}

void NetworkProcess::initializeProcessName(const AuxiliaryProcessInitializationParameters&)
{
    notImplemented();
}

void NetworkProcess::initializeSandbox(const AuxiliaryProcessInitializationParameters&, SandboxInitializationParameters&)
{
}

void NetworkProcess::platformInitializeNetworkProcess(const NetworkProcessCreationParameters& parameters)
{
#if ENABLE(SEC_ITEM_SHIM)
    // SecItemShim is needed for CFNetwork APIs that query Keychains beneath us.
    initializeSecItemShim(*this);
#endif
    platformInitializeNetworkProcessCocoa(parameters);
}

void NetworkProcess::platformTerminate()
{
    notImplemented();
}

static bool disableServiceWorkerEntitlementTestingOverride;

bool NetworkProcess::parentProcessHasServiceWorkerEntitlement() const
{
    if (disableServiceWorkerEntitlementTestingOverride)
        return false;

    static bool hasEntitlement = WTF::hasEntitlement(parentProcessConnection()->xpcConnection(), "com.apple.developer.WebKit.ServiceWorkers"_s) || WTF::hasEntitlement(parentProcessConnection()->xpcConnection(), "com.apple.developer.web-browser"_s);
    return hasEntitlement;
}

void NetworkProcess::disableServiceWorkerEntitlement()
{
    disableServiceWorkerEntitlementTestingOverride = true;
}

void NetworkProcess::clearServiceWorkerEntitlementOverride(CompletionHandler<void()>&& completionHandler)
{
    disableServiceWorkerEntitlementTestingOverride = false;
    completionHandler();
}

#endif // !PLATFORM(MACCATALYST)

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
