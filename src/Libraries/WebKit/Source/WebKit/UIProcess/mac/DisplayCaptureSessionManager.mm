/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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
#import "DisplayCaptureSessionManager.h"

#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)

#import "APIPageConfiguration.h"
#import "Logging.h"
#import "MediaPermissionUtilities.h"
#import "WKWebViewInternal.h"
#import "WebPageProxy.h"
#import "WebProcess.h"
#import "WebProcessPool.h"
#import <WebCore/CaptureDeviceManager.h>
#import <WebCore/LocalizedStrings.h>
#import <WebCore/MockRealtimeMediaSourceCenter.h>
#import <WebCore/ScreenCaptureKitCaptureSource.h>
#import <WebCore/ScreenCaptureKitSharingSessionManager.h>
#import <WebCore/SecurityOriginData.h>
#import <wtf/BlockPtr.h>
#import <wtf/MainThread.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/URLHelpers.h>
#import <wtf/cocoa/TypeCastsCocoa.h>
#import <wtf/text/StringToIntegerConversion.h>

namespace WebKit {

#if HAVE(SCREEN_CAPTURE_KIT)
void DisplayCaptureSessionManager::alertForGetDisplayMedia(WebPageProxy& page, const WebCore::SecurityOriginData& origin, CompletionHandler<void(DisplayCaptureSessionManager::CaptureSessionType)>&& completionHandler)
{

    auto webView = page.cocoaView();
    if (!webView) {
        completionHandler(DisplayCaptureSessionManager::CaptureSessionType::None);
        return;
    }

    NSString *visibleOrigin = applicationVisibleNameFromOrigin(origin);
    if (!visibleOrigin)
        visibleOrigin = applicationVisibleName();

    NSString *alertTitle = [NSString stringWithFormat:WEB_UI_NSSTRING(@"Allow â€œ%@â€ to observe one of your windows or screens?", "Message for window and screen sharing prompt"), visibleOrigin];
    auto *allowWindowButtonString = WEB_UI_NSSTRING(@"Allow to Share Window", "Allow window button title in window and screen sharing prompt");
    auto *allowScreenButtonString = WEB_UI_NSSTRING(@"Allow to Share Screen", "Allow screen button title in window and screen sharing prompt");
    auto *doNotAllowButtonString = WEB_UI_NSSTRING_KEY(@"Donâ€™t Allow", @"Donâ€™t Allow (window and screen sharing)", "Disallow button title in window and screen sharing prompt");

    auto alert = adoptNS([[NSAlert alloc] init]);
    [alert setMessageText:alertTitle];

    auto *button = [alert addButtonWithTitle:allowWindowButtonString];
    button.keyEquivalent = @"";

    button = [alert addButtonWithTitle:allowScreenButtonString];
    button.keyEquivalent = @"";

    button = [alert addButtonWithTitle:doNotAllowButtonString];
    button.keyEquivalent = @"\E";

    [alert beginSheetModalForWindow:[webView window] completionHandler:[completionBlock = makeBlockPtr(WTFMove(completionHandler))](NSModalResponse returnCode) {
        DisplayCaptureSessionManager::CaptureSessionType result = DisplayCaptureSessionManager::CaptureSessionType::None;
        switch (returnCode) {
        case NSAlertFirstButtonReturn:
            result = DisplayCaptureSessionManager::CaptureSessionType::Window;
            break;
        case NSAlertSecondButtonReturn:
            result = DisplayCaptureSessionManager::CaptureSessionType::Screen;
            break;
        case NSAlertThirdButtonReturn:
            result = DisplayCaptureSessionManager::CaptureSessionType::None;
            break;
        }

        completionBlock(result);
    }];
}
#endif

std::optional<WebCore::CaptureDevice> DisplayCaptureSessionManager::deviceSelectedForTesting(WebCore::CaptureDevice::DeviceType deviceType, unsigned indexOfDeviceSelectedForTesting)
{
    unsigned index = 0;
    for (auto& device : WebCore::RealtimeMediaSourceCenter::singleton().displayCaptureFactory().displayCaptureDeviceManager().captureDevices()) {
        if (device.enabled() && device.type() == deviceType) {
            if (index == indexOfDeviceSelectedForTesting)
                return { device };
            ++index;
        }
    }

    return std::nullopt;
}

bool DisplayCaptureSessionManager::useMockCaptureDevices() const
{
    return m_indexOfDeviceSelectedForTesting || WebCore::MockRealtimeMediaSourceCenter::mockRealtimeMediaSourceCenterEnabled();
}

void DisplayCaptureSessionManager::showWindowPicker(const WebCore::SecurityOriginData& origin, CompletionHandler<void(std::optional<WebCore::CaptureDevice>)>&& completionHandler)
{
    if (useMockCaptureDevices()) {
        completionHandler(deviceSelectedForTesting(WebCore::CaptureDevice::DeviceType::Window, m_indexOfDeviceSelectedForTesting.value_or(0)));
        return;
    }

    completionHandler(std::nullopt);
}

void DisplayCaptureSessionManager::showScreenPicker(const WebCore::SecurityOriginData&, CompletionHandler<void(std::optional<WebCore::CaptureDevice>)>&& completionHandler)
{
    if (useMockCaptureDevices()) {
        completionHandler(deviceSelectedForTesting(WebCore::CaptureDevice::DeviceType::Screen, m_indexOfDeviceSelectedForTesting.value_or(0)));
        return;
    }

    completionHandler(std::nullopt);
}

bool DisplayCaptureSessionManager::isAvailable()
{
#if HAVE(SCREEN_CAPTURE_KIT)
    return WebCore::ScreenCaptureKitCaptureSource::isAvailable();
#else
    return false;
#endif
}

DisplayCaptureSessionManager& DisplayCaptureSessionManager::singleton()
{
    ASSERT(isMainRunLoop());
    static NeverDestroyed<DisplayCaptureSessionManager> manager;
    return manager;
}

DisplayCaptureSessionManager::DisplayCaptureSessionManager()
{
}

DisplayCaptureSessionManager::~DisplayCaptureSessionManager()
{
}

bool DisplayCaptureSessionManager::canRequestDisplayCapturePermission()
{
    if (useMockCaptureDevices())
        return m_systemCanPromptForTesting == PromptOverride::CanPrompt;

#if HAVE(SCREEN_CAPTURE_KIT)
    return WebCore::ScreenCaptureKitSharingSessionManager::useSCContentSharingPicker();
#else
    return false;
#endif
}

#if HAVE(SCREEN_CAPTURE_KIT)
static WebCore::DisplayCapturePromptType toScreenCaptureKitPromptType(UserMediaPermissionRequestProxy::UserMediaDisplayCapturePromptType promptType)
{
    if (promptType == UserMediaPermissionRequestProxy::UserMediaDisplayCapturePromptType::Screen)
        return WebCore::DisplayCapturePromptType::Screen;
    if (promptType == UserMediaPermissionRequestProxy::UserMediaDisplayCapturePromptType::Window)
        return WebCore::DisplayCapturePromptType::Window;
    if (promptType == UserMediaPermissionRequestProxy::UserMediaDisplayCapturePromptType::UserChoose)
        return WebCore::DisplayCapturePromptType::UserChoose;

    ASSERT_NOT_REACHED();
    return WebCore::DisplayCapturePromptType::Screen;
}
#endif

void DisplayCaptureSessionManager::promptForGetDisplayMedia(UserMediaPermissionRequestProxy::UserMediaDisplayCapturePromptType promptType, WebPageProxy& page, const WebCore::SecurityOriginData& origin, CompletionHandler<void(std::optional<WebCore::CaptureDevice>)>&& completionHandler)
{
    if (useMockCaptureDevices()) {
        if (promptType == UserMediaPermissionRequestProxy::UserMediaDisplayCapturePromptType::Window)
            showWindowPicker(origin, WTFMove(completionHandler));
        else
            showScreenPicker(origin, WTFMove(completionHandler));
        return;
    }

#if HAVE(SCREEN_CAPTURE_KIT)
    ASSERT(isAvailable());

    if (!isAvailable() || !completionHandler) {
        completionHandler(std::nullopt);
        return;
    }

    if (WebCore::ScreenCaptureKitSharingSessionManager::isAvailable()) {
        if (!page.preferences().useGPUProcessForDisplayCapture()) {
            WebCore::ScreenCaptureKitSharingSessionManager::singleton().promptForGetDisplayMedia(toScreenCaptureKitPromptType(promptType), WTFMove(completionHandler));
            return;
        }

        Ref gpuProcess = page.configuration().processPool().ensureGPUProcess();
        gpuProcess->updateSandboxAccess(false, false, true);
        gpuProcess->promptForGetDisplayMedia(toScreenCaptureKitPromptType(promptType), WTFMove(completionHandler));
        return;
    }

    if (promptType == UserMediaPermissionRequestProxy::UserMediaDisplayCapturePromptType::Screen) {
        showScreenPicker(origin, WTFMove(completionHandler));
        return;
    }

    if (promptType == UserMediaPermissionRequestProxy::UserMediaDisplayCapturePromptType::Window) {
        showWindowPicker(origin, WTFMove(completionHandler));
        return;
    }

    alertForGetDisplayMedia(page, origin, [this, origin, completionHandler = WTFMove(completionHandler)] (DisplayCaptureSessionManager::CaptureSessionType sessionType) mutable {
        if (sessionType == CaptureSessionType::None) {
            completionHandler(std::nullopt);
            return;
        }

        if (sessionType == CaptureSessionType::Screen)
            showScreenPicker(origin, WTFMove(completionHandler));
        else
            showWindowPicker(origin, WTFMove(completionHandler));
    });
#endif
}

void DisplayCaptureSessionManager::cancelGetDisplayMediaPrompt(WebPageProxy& page)
{
#if HAVE(SCREEN_CAPTURE_KIT)
    ASSERT(isAvailable());

    if (!isAvailable() || !WebCore::ScreenCaptureKitSharingSessionManager::isAvailable())
        return;

    if (!page.preferences().useGPUProcessForDisplayCapture()) {
        WebCore::ScreenCaptureKitSharingSessionManager::singleton().cancelGetDisplayMediaPrompt();
        return;
    }

    auto gpuProcess = page.configuration().processPool().gpuProcess();
    if (!gpuProcess)
        return;

    gpuProcess->cancelGetDisplayMediaPrompt();
#endif
}

} // namespace WebKit

#endif // PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)
