/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#pragma once

#include "APIInspectorConfiguration.h"
#include "APISecurityOrigin.h"
#include "WKPage.h"
#include "WebEvent.h"
#include "WebHitTestResultData.h"
#include <WebCore/CookieConsentDecisionResult.h>
#include <WebCore/FloatRect.h>
#include <WebCore/ModalContainerTypes.h>
#include <WebCore/PermissionState.h>
#include <WebCore/ScreenOrientationType.h>
#include <wtf/CompletionHandler.h>

#if PLATFORM(COCOA)
#include <WebCore/PlatformViewController.h>
#endif

#if PLATFORM(IOS_FAMILY)
OBJC_CLASS NSArray;
OBJC_CLASS _WKActivatedElementInfo;
OBJC_CLASS UIViewController;
#endif

#if ENABLE(WEB_AUTHN)
#include "WebAuthenticationFlags.h"
#endif

#if ENABLE(WEBXR) && PLATFORM(COCOA)
#include "PlatformXRSessionEnums.h"
#include <WebCore/PlatformXR.h>
#endif

namespace WebCore {
class RegistrableDomain;
class ResourceRequest;
class SecurityOriginData;
enum class AutoplayEvent : uint8_t;
enum class AutoplayEventFlags : uint8_t;
enum class MediaProducerMediaState : uint32_t;
struct FontAttributes;
struct WindowFeatures;
struct OrganizationStorageAccessPromptQuirk;
using MediaProducerMediaStateFlags = OptionSet<MediaProducerMediaState>;
}

namespace WebKit {
class NativeWebKeyboardEvent;
class NativeWebWheelEvent;
class UserMediaPermissionCheckProxy;
class UserMediaPermissionRequestProxy;
class WebColorPickerResultListenerProxy;
class WebFrameProxy;
class WebInspectorUIProxy;
class WebOpenPanelResultListenerProxy;
class WebPageProxy;
struct FrameInfoData;
struct NavigationActionData;
}

namespace API {

class Data;
class Dictionary;
class NavigationAction;
class Object;
class OpenPanelParameters;
class PageConfiguration;
class SecurityOrigin;
class WebAuthenticationPanel;

class UIClient {
    WTF_MAKE_TZONE_ALLOCATED(UIClient);
public:
    virtual ~UIClient() { }

    virtual void createNewPage(WebKit::WebPageProxy&, Ref<API::PageConfiguration>&&, Ref<NavigationAction>&&, CompletionHandler<void(RefPtr<WebKit::WebPageProxy>&&)>&&);
    virtual void showPage(WebKit::WebPageProxy*) { }
    virtual void fullscreenMayReturnToInline(WebKit::WebPageProxy*) { }
    virtual void didEnterFullscreen(WebKit::WebPageProxy*) { }
    virtual void didExitFullscreen(WebKit::WebPageProxy*) { }
    virtual void hasVideoInPictureInPictureDidChange(WebKit::WebPageProxy*, bool) { }
    virtual void close(WebKit::WebPageProxy*) { }

    virtual bool takeFocus(WebKit::WebPageProxy*, WKFocusDirection) { return false; }
    virtual void focus(WebKit::WebPageProxy*) { }
    virtual void unfocus(WebKit::WebPageProxy*) { }
    virtual bool focusFromServiceWorker(WebKit::WebPageProxy&) { return false; }

    virtual void runJavaScriptAlert(WebKit::WebPageProxy&, const WTF::String&, WebKit::WebFrameProxy*, WebKit::FrameInfoData&&, Function<void()>&& completionHandler) { completionHandler(); }
    virtual void runJavaScriptConfirm(WebKit::WebPageProxy&, const WTF::String&, WebKit::WebFrameProxy*, WebKit::FrameInfoData&&, Function<void(bool)>&& completionHandler) { completionHandler(false); }
    virtual void runJavaScriptPrompt(WebKit::WebPageProxy&, const WTF::String&, const WTF::String&, WebKit::WebFrameProxy*, WebKit::FrameInfoData&&, Function<void(const WTF::String&)>&& completionHandler) { completionHandler(WTF::String()); }

    virtual void setStatusText(WebKit::WebPageProxy*, const WTF::String&) { }
    virtual void mouseDidMoveOverElement(WebKit::WebPageProxy&, const WebKit::WebHitTestResultData&, OptionSet<WebKit::WebEventModifier>, Object*) { }

    virtual void didNotHandleKeyEvent(WebKit::WebPageProxy*, const WebKit::NativeWebKeyboardEvent&) { }
    virtual void didNotHandleWheelEvent(WebKit::WebPageProxy*, const WebKit::NativeWebWheelEvent&) { }

    virtual void toolbarsAreVisible(WebKit::WebPageProxy&, Function<void(bool)>&& completionHandler) { completionHandler(true); }
    virtual void setToolbarsAreVisible(WebKit::WebPageProxy&, bool) { }
    virtual void menuBarIsVisible(WebKit::WebPageProxy&, Function<void(bool)>&& completionHandler) { completionHandler(true); }
    virtual void setMenuBarIsVisible(WebKit::WebPageProxy&, bool) { }
    virtual void statusBarIsVisible(WebKit::WebPageProxy&, Function<void(bool)>&& completionHandler) { completionHandler(true); }
    virtual void setStatusBarIsVisible(WebKit::WebPageProxy&, bool) { }
    virtual void setIsResizable(WebKit::WebPageProxy&, bool) { }

    virtual void setWindowFrame(WebKit::WebPageProxy&, const WebCore::FloatRect&) { }
    virtual void windowFrame(WebKit::WebPageProxy&, Function<void(WebCore::FloatRect)>&& completionHandler) { completionHandler({ }); }

    virtual bool canRunBeforeUnloadConfirmPanel() const { return false; }
    virtual void runBeforeUnloadConfirmPanel(WebKit::WebPageProxy&, const WTF::String&, WebKit::WebFrameProxy*, WebKit::FrameInfoData&&, Function<void(bool)>&& completionHandler) { completionHandler(true); }

    virtual void pageDidScroll(WebKit::WebPageProxy*) { }

    virtual void exceededDatabaseQuota(WebKit::WebPageProxy*, WebKit::WebFrameProxy*, SecurityOrigin*, const WTF::String&, const WTF::String&, unsigned long long currentQuota, unsigned long long, unsigned long long, unsigned long long, Function<void (unsigned long long)>&& completionHandler)
    {
        completionHandler(currentQuota);
    }

    virtual bool lockScreenOrientation(WebKit::WebPageProxy&, WebCore::ScreenOrientationType) { return false; }
    virtual void unlockScreenOrientation(WebKit::WebPageProxy&) { }

    virtual bool needsFontAttributes() const { return false; }
    virtual void didChangeFontAttributes(const WebCore::FontAttributes&) { }

    virtual bool runOpenPanel(WebKit::WebPageProxy&, WebKit::WebFrameProxy*, WebKit::FrameInfoData&&, OpenPanelParameters*, WebKit::WebOpenPanelResultListenerProxy*) { return false; }
    virtual void decidePolicyForGeolocationPermissionRequest(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, const WebKit::FrameInfoData&, Function<void(bool)>&) { }
    virtual void decidePolicyForUserMediaPermissionRequest(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, SecurityOrigin&, SecurityOrigin&, WebKit::UserMediaPermissionRequestProxy&);
    virtual void decidePolicyForScreenCaptureUnmuting(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, SecurityOrigin&, SecurityOrigin&, CompletionHandler<void(bool isAllowed)>&& completionHandler) { completionHandler(false); }
    virtual void checkUserMediaPermissionForOrigin(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, SecurityOrigin&, SecurityOrigin&, WebKit::UserMediaPermissionCheckProxy&);
    virtual void decidePolicyForNotificationPermissionRequest(WebKit::WebPageProxy&, SecurityOrigin&, CompletionHandler<void(bool allowed)>&& completionHandler) { completionHandler(false); }
    virtual void requestStorageAccessConfirm(WebKit::WebPageProxy&, WebKit::WebFrameProxy*, const WebCore::RegistrableDomain& requestingDomain, const WebCore::RegistrableDomain& currentDomain, std::optional<WebCore::OrganizationStorageAccessPromptQuirk>&&, CompletionHandler<void(bool)>&& completionHandler) { completionHandler(true); }
    virtual void requestCookieConsent(CompletionHandler<void(WebCore::CookieConsentDecisionResult)>&& completionHandler) { completionHandler(WebCore::CookieConsentDecisionResult::NotSupported); }

    // Printing.
    virtual float headerHeight(WebKit::WebPageProxy&, WebKit::WebFrameProxy&) { return 0; }
    virtual float footerHeight(WebKit::WebPageProxy&, WebKit::WebFrameProxy&) { return 0; }
    virtual void drawHeader(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, WebCore::FloatRect&&) { }
    virtual void drawFooter(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, WebCore::FloatRect&&) { }
    virtual void printFrame(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, const WebCore::FloatSize& pdfFirstPageSize, CompletionHandler<void()>&& completionHandler) { completionHandler(); }

    virtual bool canRunModal() const { return false; }
    virtual void runModal(WebKit::WebPageProxy&) { }

    virtual void saveDataToFileInDownloadsFolder(WebKit::WebPageProxy*, const WTF::String&, const WTF::String&, const WTF::URL&, Data&) { }

    virtual void pinnedStateDidChange(WebKit::WebPageProxy&) { }

    virtual void isPlayingMediaDidChange(WebKit::WebPageProxy&) { }
    virtual void mediaCaptureStateDidChange(WebCore::MediaProducerMediaStateFlags) { }
    virtual void handleAutoplayEvent(WebKit::WebPageProxy&, WebCore::AutoplayEvent, OptionSet<WebCore::AutoplayEventFlags>) { }

#if PLATFORM(IOS_FAMILY)
#if HAVE(APP_LINKS)
    virtual bool shouldIncludeAppLinkActionsForElement(_WKActivatedElementInfo *) { return true; }
#endif
    virtual RetainPtr<NSArray> actionsForElement(_WKActivatedElementInfo *, RetainPtr<NSArray> defaultActions) { return defaultActions; }
    virtual void didNotHandleTapAsClick(const WebCore::IntPoint&) { }
    virtual void statusBarWasTapped() { }
    virtual bool setShouldKeepScreenAwake(bool) { return false; }
#endif
#if PLATFORM(COCOA)
    virtual PlatformViewController *presentingViewController() { return nullptr; }
    virtual std::optional<double> dataDetectionReferenceDate() { return std::nullopt; }
#endif

#if ENABLE(POINTER_LOCK)
    virtual void requestPointerLock(WebKit::WebPageProxy*) { }
    virtual void didLosePointerLock(WebKit::WebPageProxy*) { }
#endif

#if ENABLE(DEVICE_ORIENTATION)
    virtual void shouldAllowDeviceOrientationAndMotionAccess(WebKit::WebPageProxy&, WebKit::WebFrameProxy& webFrameProxy, WebKit::FrameInfoData&&, CompletionHandler<void(bool)>&& completionHandler) { completionHandler(false); }
#endif

    virtual void didClickAutoFillButton(WebKit::WebPageProxy&, Object*) { }

    virtual void didResignInputElementStrongPasswordAppearance(WebKit::WebPageProxy&, Object*) { }

    virtual void imageOrMediaDocumentSizeChanged(const WebCore::IntSize&) { }

    virtual void didShowSafeBrowsingWarning() { }

    virtual void confirmPDFOpening(WebKit::WebPageProxy&, const WTF::URL&, WebKit::FrameInfoData&&, CompletionHandler<void(bool)>&& completionHandler) { completionHandler(true); }

#if ENABLE(WEB_AUTHN)
    virtual void runWebAuthenticationPanel(WebKit::WebPageProxy&, WebAuthenticationPanel&, WebKit::WebFrameProxy&, WebKit::FrameInfoData&&, CompletionHandler<void(WebKit::WebAuthenticationPanelResult)>&& completionHandler) { completionHandler(WebKit::WebAuthenticationPanelResult::Unavailable); }

    virtual void requestWebAuthenticationConditonalMediationRegistration(const WTF::String&, CompletionHandler<void(std::optional<bool>)>&& completionHandler)
    {
        completionHandler(std::nullopt);
    }
#endif

    virtual void didAttachLocalInspector(WebKit::WebPageProxy&, WebKit::WebInspectorUIProxy&) { }
    virtual void willCloseLocalInspector(WebKit::WebPageProxy&, WebKit::WebInspectorUIProxy&) { }
    virtual Ref<InspectorConfiguration> configurationForLocalInspector(WebKit::WebPageProxy&, WebKit::WebInspectorUIProxy&)
    {
        return InspectorConfiguration::create();
    }
    virtual void didEnableInspectorBrowserDomain(WebKit::WebPageProxy&) { }
    virtual void didDisableInspectorBrowserDomain(WebKit::WebPageProxy&) { }

    virtual void decidePolicyForMediaKeySystemPermissionRequest(WebKit::WebPageProxy&, SecurityOrigin&, const WTF::String& keySystem, CompletionHandler<void(bool)>&&);

    virtual void queryPermission(const WTF::String& permissionName, SecurityOrigin& origin, CompletionHandler<void(std::optional<WebCore::PermissionState>)>&& completionHandler) { completionHandler({ }); }

#if ENABLE(WEBXR) && PLATFORM(COCOA)
    virtual void requestPermissionOnXRSessionFeatures(WebKit::WebPageProxy&, const WebCore::SecurityOriginData&, PlatformXR::SessionMode, const PlatformXR::Device::FeatureList& granted, const PlatformXR::Device::FeatureList& /* consentRequired */, const PlatformXR::Device::FeatureList& /* consentOptional */, const PlatformXR::Device::FeatureList& /* requiredFeaturesRequested */, const PlatformXR::Device::FeatureList& /* optionalFeaturesRequested */, CompletionHandler<void(std::optional<PlatformXR::Device::FeatureList>&&)>&& completionHandler) { completionHandler(granted); }
    virtual void supportedXRSessionFeatures(PlatformXR::Device::FeatureList& vrFeatures, PlatformXR::Device::FeatureList& arFeatures) { }

#if PLATFORM(IOS_FAMILY)
    virtual void startXRSession(WebKit::WebPageProxy&, const PlatformXR::Device::FeatureList&, CompletionHandler<void(RetainPtr<id>, PlatformViewController *)>&& completionHandler) { completionHandler(nil, nil); }
    virtual void endXRSession(WebKit::WebPageProxy&, WebKit::PlatformXRSessionEndReason) { }
#endif
#endif

    virtual void updateAppBadge(WebKit::WebPageProxy&, const WebCore::SecurityOriginData&, std::optional<uint64_t>) { }
    virtual void updateClientBadge(WebKit::WebPageProxy&, const WebCore::SecurityOriginData&, std::optional<uint64_t>) { }

    virtual void didAdjustVisibilityWithSelectors(WebKit::WebPageProxy&, Vector<WTF::String>&&) { }

#if ENABLE(GAMEPAD)
    virtual void recentlyAccessedGamepadsForTesting(WebKit::WebPageProxy&) { }
    virtual void stoppedAccessingGamepadsForTesting(WebKit::WebPageProxy&) { }
#endif
};

} // namespace API
