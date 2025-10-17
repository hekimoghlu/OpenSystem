/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 20, 2023.
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

#include "DrawingAreaInfo.h"
#include "FrameTreeCreationParameters.h"
#include "LayerTreeContext.h"
#include "SandboxExtension.h"
#include "SessionState.h"
#include "UserContentControllerParameters.h"
#include "ViewWindowCoordinates.h"
#include "VisitedLinkTableIdentifier.h"
#include "WebPageGroupData.h"
#include "WebPageProxyIdentifier.h"
#include "WebPreferencesStore.h"
#include "WebURLSchemeHandlerIdentifier.h"
#include "WebsitePoliciesData.h"
#include <WebCore/ActivityState.h>
#include <WebCore/Color.h>
#include <WebCore/ContentSecurityPolicy.h>
#include <WebCore/DestinationColorSpace.h>
#include <WebCore/FloatSize.h>
#include <WebCore/FrameIdentifier.h>
#include <WebCore/HighlightVisibility.h>
#include <WebCore/IntDegrees.h>
#include <WebCore/IntSize.h>
#include <WebCore/LayerHostingContextIdentifier.h>
#include <WebCore/LayoutMilestone.h>
#include <WebCore/MediaProducer.h>
#include <WebCore/PageIdentifier.h>
#include <WebCore/Pagination.h>
#include <WebCore/ScrollTypes.h>
#include <WebCore/ShouldRelaxThirdPartyCookieBlocking.h>
#include <WebCore/UserInterfaceLayoutDirection.h>
#include <WebCore/ViewportArguments.h>
#include <WebCore/WindowFeatures.h>
#include <wtf/RobinHoodHashSet.h>
#include <wtf/text/WTFString.h>

#if ENABLE(APPLICATION_MANIFEST)
#include <WebCore/ApplicationManifest.h>
#endif

#if ENABLE(ADVANCED_PRIVACY_PROTECTIONS)
#include <WebCore/LinkDecorationFilteringData.h>
#endif

#if ENABLE(WK_WEB_EXTENSIONS)
#include "WebExtensionControllerParameters.h"
#endif

#if (PLATFORM(GTK) || PLATFORM(WPE)) && USE(GBM)
#include "DMABufRendererBufferFormat.h"
#endif

#if PLATFORM(IOS_FAMILY)
#include "HardwareKeyboardState.h"
#endif

#if PLATFORM(VISION) && ENABLE(GAMEPAD)
#include <WebCore/ShouldRequireExplicitConsentForGamepadAccess.h>
#endif

#if HAVE(AUDIT_TOKEN)
#include "CoreIPCAuditToken.h"
#endif

namespace WebCore {
enum class SandboxFlag : uint16_t;
using SandboxFlags = OptionSet<SandboxFlag>;
}

namespace WebKit {

struct RemotePageParameters {
    URL initialMainDocumentURL;
    FrameTreeCreationParameters frameTreeParameters;
    std::optional<WebsitePoliciesData> websitePoliciesData;
};

struct WebPageCreationParameters {
    WebCore::IntSize viewSize { };

    OptionSet<WebCore::ActivityState> activityState { };
    
    WebPreferencesStore store { };
    DrawingAreaType drawingAreaType { };
    DrawingAreaIdentifier drawingAreaIdentifier;
    WebPageProxyIdentifier webPageProxyIdentifier;
    WebPageGroupData pageGroupData;

    bool isEditable { false };

    WebCore::Color underlayColor { };

    bool useFixedLayout { false };
    WebCore::IntSize fixedLayoutSize { };

    WebCore::FloatSize defaultUnobscuredSize { };
    WebCore::FloatSize minimumUnobscuredSize { };
    WebCore::FloatSize maximumUnobscuredSize { };

    std::optional<WebCore::FloatRect> viewExposedRect { };

    std::optional<uint32_t> displayID { };
    std::optional<unsigned> nominalFramesPerSecond { };

    bool alwaysShowsHorizontalScroller { false };
    bool alwaysShowsVerticalScroller { false };

    bool suppressScrollbarAnimations { false };

    WebCore::Pagination::Mode paginationMode { WebCore::Pagination::Mode::Unpaginated };
    bool paginationBehavesLikeColumns { false };
    double pageLength { 0 };
    double gapBetweenPages { 0 };
    
    String userAgent { };

    VisitedLinkTableIdentifier visitedLinkTableID;
    bool canRunBeforeUnloadConfirmPanel { false };
    bool canRunModal { false };

    float deviceScaleFactor { 0 };
#if USE(GRAPHICS_LAYER_WC) || USE(GRAPHICS_LAYER_TEXTURE_MAPPER)
    float intrinsicDeviceScaleFactor { 0 };
#endif
    float viewScaleFactor { 0 };

    double textZoomFactor { 1 };
    double pageZoomFactor { 1 };

    float topContentInset { 0 };
    
    float mediaVolume { 0 };
    WebCore::MediaProducerMutedStateFlags muted { };
    bool openedByDOM { false };
    bool mayStartMediaWhenInWindow { false };
    bool mediaPlaybackIsSuspended { false };

    WebCore::IntSize minimumSizeForAutoLayout { };
    WebCore::IntSize sizeToContentAutoSizeMaximumSize { };
    bool autoSizingShouldExpandToViewHeight { false };
    std::optional<WebCore::FloatSize> viewportSizeForCSSViewportUnits { };
    
    WebCore::ScrollPinningBehavior scrollPinningBehavior { WebCore::ScrollPinningBehavior::DoNotPin };

    // FIXME: This should be std::optional<WebCore::ScrollbarOverlayStyle>, but we would need to
    // correctly handle enums inside Optionals when encoding and decoding. 
    std::optional<uint32_t> scrollbarOverlayStyle { };

    bool backgroundExtendsBeyondPage { false };

    LayerHostingMode layerHostingMode { LayerHostingMode::InProcess };

    bool hasResourceLoadClient { false };

    Vector<String> mimeTypesWithCustomContentProviders { };

    bool controlledByAutomation { false };
    bool isProcessSwap { false };

    bool useDarkAppearance { false };
    bool useElevatedUserInterfaceLevel { false };

#if PLATFORM(MAC)
    std::optional<WebCore::DestinationColorSpace> colorSpace { };
    bool useFormSemanticContext { false };
    int headerBannerHeight { 0 };
    int footerBannerHeight { 0 };
    std::optional<ViewWindowCoordinates> viewWindowCoordinates { };
#endif
#if ENABLE(META_VIEWPORT)
    bool ignoresViewportScaleLimits { false };
    WebCore::FloatSize viewportConfigurationViewLayoutSize { };
    double viewportConfigurationLayoutSizeScaleFactorFromClient { 0 };
    double viewportConfigurationMinimumEffectiveDeviceWidth { 0 };
    WebCore::FloatSize viewportConfigurationViewSize { };
    std::optional<WebCore::ViewportArguments> overrideViewportArguments { };
#endif
#if PLATFORM(IOS_FAMILY)
    WebCore::FloatSize screenSize { };
    WebCore::FloatSize availableScreenSize { };
    WebCore::FloatSize overrideScreenSize { };
    WebCore::FloatSize overrideAvailableScreenSize { };
    float textAutosizingWidth { 0 };
    WebCore::IntDegrees deviceOrientation { 0 };
    HardwareKeyboardState hardwareKeyboardState { };
    bool canShowWhileLocked { false };
    bool isCapturingScreen { false };
    WebCore::Color insertionPointColor { };
#endif
#if PLATFORM(COCOA)
    bool smartInsertDeleteEnabled { false };
    Vector<String> additionalSupportedImageTypes { };
    Vector<SandboxExtension::Handle> gpuIOKitExtensionHandles { };
    Vector<SandboxExtension::Handle> gpuMachExtensionHandles { };
#endif
#if PLATFORM(MAC)
    SandboxExtension::Handle renderServerMachExtensionHandle { };
#endif
#if HAVE(STATIC_FONT_REGISTRY)
    Vector<SandboxExtension::Handle> fontMachExtensionHandles { };
#endif
#if HAVE(HOSTED_CORE_ANIMATION)
    WTF::MachSendRight acceleratedCompositingPort { };
#endif
#if HAVE(APP_ACCENT_COLORS)
    WebCore::Color accentColor { };
#if PLATFORM(MAC)
    bool appUsesCustomAccentColor { false };
#endif
#endif
#if USE(WPE_RENDERER)
    UnixFileDescriptor hostFileDescriptor { };
#endif
#if USE(GRAPHICS_LAYER_TEXTURE_MAPPER) || USE(GRAPHICS_LAYER_WC)
    uint64_t nativeWindowHandle { 0 };
#endif
#if USE(GRAPHICS_LAYER_WC)
    bool usesOffscreenRendering { false };
#endif
    bool shouldScaleViewToFitDocument { false };

    WebCore::UserInterfaceLayoutDirection userInterfaceLayoutDirection { WebCore::UserInterfaceLayoutDirection::LTR };
    OptionSet<WebCore::LayoutMilestone> observedLayoutMilestones { };

    String overrideContentSecurityPolicy { };
    std::optional<double> cpuLimit { };

    HashMap<String, WebURLSchemeHandlerIdentifier> urlSchemeHandlers { };
    Vector<String> urlSchemesWithLegacyCustomProtocolHandlers { };

#if ENABLE(APPLICATION_MANIFEST)
    std::optional<WebCore::ApplicationManifest> applicationManifest { };
#endif

    bool needsFontAttributes { false };

    // WebRTC members.
    bool iceCandidateFilteringEnabled { true };
    bool enumeratingAllNetworkInterfacesEnabled { false };

    UserContentControllerParameters userContentControllerParameters;

#if ENABLE(WK_WEB_EXTENSIONS)
    std::optional<WebExtensionControllerParameters> webExtensionControllerParameters { };
#endif

    std::optional<WebCore::Color> backgroundColor { };

    std::optional<WebCore::PageIdentifier> oldPageID { };

    String overriddenMediaType { };
    Vector<String> corsDisablingPatterns { };
    HashSet<String> maskedURLSchemes { };
    bool loadsSubresources { true };
    std::optional<MemoryCompactLookupOnlyRobinHoodHashSet<String>> allowedNetworkHosts { };
    std::optional<std::pair<uint16_t, uint16_t>> portsForUpgradingInsecureSchemeForTesting { };

    bool crossOriginAccessControlCheckEnabled { true };
    String processDisplayName { };

    bool shouldCaptureAudioInUIProcess { false };
    bool shouldCaptureAudioInGPUProcess { false };
    bool shouldCaptureVideoInUIProcess { false };
    bool shouldCaptureVideoInGPUProcess { false };
    bool shouldCaptureDisplayInUIProcess { false };
    bool shouldCaptureDisplayInGPUProcess { false };
    bool shouldRenderCanvasInGPUProcess { false };
    bool shouldRenderDOMInGPUProcess { false };
    bool shouldPlayMediaInGPUProcess { false };
#if ENABLE(WEBGL)
    bool shouldRenderWebGLInGPUProcess { false };
#endif
    bool shouldEnableVP8Decoder { false };
    bool shouldEnableVP9Decoder { false };
#if ENABLE(APP_BOUND_DOMAINS)
    bool limitsNavigationsToAppBoundDomains { false };
#endif
    bool lastNavigationWasAppInitiated { true };
    bool canUseCredentialStorage { true };

    WebCore::ShouldRelaxThirdPartyCookieBlocking shouldRelaxThirdPartyCookieBlocking { WebCore::ShouldRelaxThirdPartyCookieBlocking::No };
    
    bool httpsUpgradeEnabled { true };

#if PLATFORM(IOS) || PLATFORM(VISION)
    bool allowsDeprecatedSynchronousXMLHttpRequestDuringUnload { false };
#endif
    
#if ENABLE(APP_HIGHLIGHTS)
    WebCore::HighlightVisibility appHighlightsVisible { WebCore::HighlightVisibility::Hidden };
#endif

#if HAVE(TOUCH_BAR)
    bool requiresUserActionForEditingControlsManager { false };
#endif

    bool hasResizableWindows { false };

    WebCore::ContentSecurityPolicyModeForExtension contentSecurityPolicyModeForExtension { WebCore::ContentSecurityPolicyModeForExtension::None };

    std::optional<RemotePageParameters> remotePageParameters { };
    WebCore::FrameIdentifier mainFrameIdentifier;
    String openedMainFrameName;
    std::optional<WebCore::FrameIdentifier> mainFrameOpenerIdentifier { };
    WebCore::SandboxFlags initialSandboxFlags;
    std::optional<WebCore::WindowFeatures> windowFeatures { };

#if ENABLE(ADVANCED_PRIVACY_PROTECTIONS)
    Vector<WebCore::LinkDecorationFilteringData> linkDecorationFilteringData { };
    Vector<WebCore::LinkDecorationFilteringData> allowedQueryParametersForAdvancedPrivacyProtections { };
#endif

#if HAVE(MACH_BOOTSTRAP_EXTENSION)
    SandboxExtension::Handle machBootstrapHandle { };
#endif

#if PLATFORM(GTK) || PLATFORM(WPE)
#if USE(GBM)
    Vector<DMABufRendererBufferFormat> preferredBufferFormats { };
#endif
#endif

#if PLATFORM(VISION) && ENABLE(GAMEPAD)
    WebCore::ShouldRequireExplicitConsentForGamepadAccess gamepadAccessRequiresExplicitConsent { WebCore::ShouldRequireExplicitConsentForGamepadAccess::No };
#endif

#if HAVE(AUDIT_TOKEN)
    std::optional<CoreIPCAuditToken> presentingApplicationAuditToken;
#endif

#if PLATFORM(COCOA)
    String presentingApplicationBundleIdentifier;
#endif
};

} // namespace WebKit
