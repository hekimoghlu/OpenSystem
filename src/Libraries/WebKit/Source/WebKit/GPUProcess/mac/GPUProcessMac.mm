/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 23, 2023.
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
#import "GPUProcess.h"

#if ENABLE(GPU_PROCESS) && (PLATFORM(MAC) || PLATFORM(MACCATALYST))

#import "GPUProcessCreationParameters.h"
#import "SandboxInitializationParameters.h"
#import "WKFoundation.h"
#import <WebCore/LocalizedStrings.h>
#import <WebCore/PlatformScreen.h>
#import <WebCore/ScreenProperties.h>
#import <WebCore/WebMAudioUtilitiesCocoa.h>
#import <pal/spi/cocoa/CoreServicesSPI.h>
#import <pal/spi/cocoa/LaunchServicesSPI.h>
#import <sysexits.h>
#import <wtf/BlockPtr.h>
#import <wtf/MemoryPressureHandler.h>
#import <wtf/ProcessPrivilege.h>
#import <wtf/text/WTFString.h>

namespace WebKit {
using namespace WebCore;

void GPUProcess::initializeProcess(const AuxiliaryProcessInitializationParameters&)
{
    setApplicationIsDaemon();

#if HAVE(CSCHECKFIXDISABLE)
    _CSCheckFixDisable();
#endif
}

void GPUProcess::initializeProcessName(const AuxiliaryProcessInitializationParameters& parameters)
{
#if PLATFORM(MAC)
    m_uiProcessName = parameters.uiProcessName;
#endif
}

#if PLATFORM(MAC)
void GPUProcess::updateProcessName()
{
#if !PLATFORM(MACCATALYST)
    NSString *applicationName = [NSString stringWithFormat:WEB_UI_NSSTRING(@"%@ Graphics and Media", "visible name of the GPU process. The argument is the application name."), (NSString *)m_uiProcessName];
    auto result = _LSSetApplicationInformationItem(kLSDefaultSessionID, _LSGetCurrentApplicationASN(), _kLSDisplayNameKey, (CFStringRef)applicationName, nullptr);
    ASSERT_UNUSED(result, result == noErr);
#endif
}
#endif

void GPUProcess::initializeSandbox(const AuxiliaryProcessInitializationParameters& parameters, SandboxInitializationParameters& sandboxParameters)
{
    // Need to overide the default, because service has a different bundle ID.
    NSBundle *webKit2Bundle = [NSBundle bundleForClass:NSClassFromString(@"WKWebView")];

    sandboxParameters.setOverrideSandboxProfilePath([webKit2Bundle pathForResource:@"com.apple.WebKit.GPUProcess" ofType:@"sb"]);

    AuxiliaryProcess::initializeSandbox(parameters, sandboxParameters);
}

#if PLATFORM(MAC)
void GPUProcess::setScreenProperties(const WebCore::ScreenProperties& screenProperties)
{
#if !HAVE(AVPLAYER_VIDEORANGEOVERRIDE)
    // Only override HDR support at the MediaToolbox level if AVPlayer.videoRangeOverride support is
    // not present, as the MediaToolbox override functionality is both duplicative and process global.

    // This override is not necessary if AVFoundation is allowed to communicate
    // with the window server to query for HDR support.
    if (hasProcessPrivilege(ProcessPrivilege::CanCommunicateWithWindowServer)) {
        setShouldOverrideScreenSupportsHighDynamicRange(false, false);
        return;
    }

    bool allScreensAreHDR = allOf(screenProperties.screenDataMap.values(), [] (auto& screenData) {
        return screenData.screenSupportsHighDynamicRange;
    });
    setShouldOverrideScreenSupportsHighDynamicRange(true, allScreensAreHDR);
#endif
}

void GPUProcess::openDirectoryCacheInvalidated(SandboxExtension::Handle&& handle)
{
    auto cacheInvalidationHandler = [handle = WTFMove(handle)] () mutable {
        AuxiliaryProcess::openDirectoryCacheInvalidated(WTFMove(handle));
    };
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), makeBlockPtr(WTFMove(cacheInvalidationHandler)).get());
}
#endif // PLATFORM(MAC)

#if HAVE(POWERLOG_TASK_MODE_QUERY)
void GPUProcess::enablePowerLogging(SandboxExtension::Handle&& handle)
{
    SandboxExtension::consumePermanently(WTFMove(handle));
}
#endif // HAVE(POWERLOG_TASK_MODE_QUERY)

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && (PLATFORM(MAC) || PLATFORM(MACCATALYST))
