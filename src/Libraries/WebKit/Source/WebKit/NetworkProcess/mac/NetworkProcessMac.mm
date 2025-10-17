/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 9, 2022.
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

#if PLATFORM(MAC) || PLATFORM(MACCATALYST)

#import "NetworkCache.h"
#import "NetworkProcessCreationParameters.h"
#import "NetworkResourceLoader.h"
#import "SandboxExtension.h"
#import "SandboxInitializationParameters.h"
#import "SecItemShim.h"
#import "WKFoundation.h"
#import <WebCore/CertificateInfo.h>
#import <WebCore/LocalizedStrings.h>
#import <notify.h>
#import <pal/spi/cf/CFNetworkSPI.h>
#import <pal/spi/cocoa/LaunchServicesSPI.h>
#import <pal/spi/mac/HIServicesSPI.h>
#import <sysexits.h>
#import <wtf/FileSystem.h>
#import <wtf/MemoryPressureHandler.h>
#import <wtf/text/MakeString.h>
#import <wtf/text/WTFString.h>

namespace WebKit {

void NetworkProcess::initializeProcess(const AuxiliaryProcessInitializationParameters&)
{
    setApplicationIsDaemon();
    launchServicesCheckIn();
}

void NetworkProcess::initializeProcessName(const AuxiliaryProcessInitializationParameters& parameters)
{
#if !PLATFORM(MACCATALYST)
    NSString *applicationName = [NSString stringWithFormat:WEB_UI_NSSTRING(@"%@ Networking", "visible name of the network process. The argument is the application name."), (NSString *)parameters.uiProcessName];
    _LSSetApplicationInformationItem(kLSDefaultSessionID, _LSGetCurrentApplicationASN(), _kLSDisplayNameKey, (CFStringRef)applicationName, nullptr);
#endif
}

void NetworkProcess::platformInitializeNetworkProcess(const NetworkProcessCreationParameters& parameters)
{
    platformInitializeNetworkProcessCocoa(parameters);

#if ENABLE(SEC_ITEM_SHIM)
    // SecItemShim is needed for CFNetwork APIs that query Keychains beneath us.
    initializeSecItemShim(*this);
#endif
}

void NetworkProcess::initializeSandbox(const AuxiliaryProcessInitializationParameters& parameters, SandboxInitializationParameters& sandboxParameters)
{
    auto webKitBundle = [NSBundle bundleForClass:NSClassFromString(@"WKWebView")];

    sandboxParameters.setOverrideSandboxProfilePath(makeString(String([webKitBundle resourcePath]), "/com.apple.WebKit.NetworkProcess.sb"_s));

    AuxiliaryProcess::initializeSandbox(parameters, sandboxParameters);
}

void NetworkProcess::platformTerminate()
{
    if (m_clearCacheDispatchGroup) {
        dispatch_group_wait(m_clearCacheDispatchGroup.get(), DISPATCH_TIME_FOREVER);
        m_clearCacheDispatchGroup = nullptr;
    }
}

#if PLATFORM(MACCATALYST)
bool NetworkProcess::parentProcessHasServiceWorkerEntitlement() const
{
    return true;
}
#endif

} // namespace WebKit

#endif // PLATFORM(MAC)
