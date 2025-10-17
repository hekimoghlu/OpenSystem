/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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

#include "APIData.h"
#include "AccessibilityPreferences.h"
#include "AuxiliaryProcessCreationParameters.h"
#include "CacheModel.h"
#include "SandboxExtension.h"
#include "ScriptTelemetry.h"
#include "TextCheckerState.h"
#include "UserData.h"

#include "WebProcessDataStoreParameters.h"
#include <WebCore/CrossOriginMode.h>
#include <wtf/HashMap.h>
#include <wtf/OptionSet.h>
#include <wtf/ProcessID.h>
#include <wtf/RetainPtr.h>
#include <wtf/Vector.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA) || PLATFORM(GTK) || (PLATFORM(WPE) && ENABLE(WPE_PLATFORM))
#include <WebCore/ScreenProperties.h>
#endif

#if PLATFORM(COCOA)
#include <WebCore/PlatformScreen.h>
#include <wtf/MachSendRight.h>
#endif

#if PLATFORM(IOS_FAMILY)
#include <WebCore/RenderThemeIOS.h>
#include <pal/system/ios/UserInterfaceIdiom.h>
#endif

#if PLATFORM(GTK) || PLATFORM(WPE)
#include "RendererBufferTransportMode.h"
#include <WebCore/SystemSettings.h>
#include <wtf/MemoryPressureHandler.h>
#endif

namespace API {
class Data;
}

namespace WebKit {

struct WebProcessCreationParameters {
    AuxiliaryProcessCreationParameters auxiliaryProcessParameters;
    String injectedBundlePath;
    SandboxExtension::Handle injectedBundlePathExtensionHandle;
    Vector<SandboxExtension::Handle> additionalSandboxExtensionHandles;

    UserData initializationUserData;

#if PLATFORM(COCOA) && ENABLE(REMOTE_INSPECTOR)
    Vector<SandboxExtension::Handle> enableRemoteWebInspectorExtensionHandles;
#endif

    Vector<String> urlSchemesRegisteredAsEmptyDocument;
    Vector<String> urlSchemesRegisteredAsSecure;
    Vector<String> urlSchemesRegisteredAsBypassingContentSecurityPolicy;
    Vector<String> urlSchemesForWhichDomainRelaxationIsForbidden;
    Vector<String> urlSchemesRegisteredAsLocal;
    Vector<String> urlSchemesRegisteredAsNoAccess;
    Vector<String> urlSchemesRegisteredAsDisplayIsolated;
    Vector<String> urlSchemesRegisteredAsCORSEnabled;
    Vector<String> urlSchemesRegisteredAsAlwaysRevalidated;
    Vector<String> urlSchemesRegisteredAsCachePartitioned;
    Vector<String> urlSchemesRegisteredAsCanDisplayOnlyIfCanRequest;

#if ENABLE(WK_WEB_EXTENSIONS)
    Vector<String> urlSchemesRegisteredAsWebExtensions;
#endif

    Vector<String> fontAllowList;
    Vector<String> overrideLanguages;
#if USE(GSTREAMER)
    Vector<String> gstreamerOptions;
#endif

    CacheModel cacheModel;

    double defaultRequestTimeoutInterval { INT_MAX };
    unsigned backForwardCacheCapacity { 0 };

    bool shouldAlwaysUseComplexTextCodePath { false };
    bool shouldEnableMemoryPressureReliefLogging { false };
    bool shouldSuppressMemoryPressureHandler { false };
    bool disableFontSubpixelAntialiasingForTesting { false };
    bool fullKeyboardAccessEnabled { false };
#if HAVE(MOUSE_DEVICE_OBSERVATION)
    bool hasMouseDevice { false };
#endif
#if HAVE(STYLUS_DEVICE_OBSERVATION)
    bool hasStylusDevice { false };
#endif
    bool memoryCacheDisabled { false };
    bool attrStyleEnabled { false };
    bool shouldThrowExceptionForGlobalConstantRedeclaration { true };
    WebCore::CrossOriginMode crossOriginMode { WebCore::CrossOriginMode::Shared }; // Cross-origin isolation via COOP+COEP headers.

#if ENABLE(SERVICE_CONTROLS)
    bool hasImageServices { false };
    bool hasSelectionServices { false };
    bool hasRichContentServices { false };
#endif

    OptionSet<TextCheckerState> textCheckerState;

#if PLATFORM(COCOA)
    String uiProcessBundleIdentifier;
    int latencyQOS { 0 };
    int throughputQOS { 0 };
#endif

    ProcessID presentingApplicationPID { 0 };

#if PLATFORM(COCOA)
    String uiProcessBundleResourcePath;
    SandboxExtension::Handle uiProcessBundleResourcePathExtensionHandle;

    bool shouldEnableJIT { false };
    bool shouldEnableFTLJIT { false };
    bool accessibilityEnhancedUserInterfaceEnabled { false };
    
    RefPtr<API::Data> bundleParameterData;
#endif // PLATFORM(COCOA)

#if ENABLE(NOTIFICATIONS)
    HashMap<String, bool> notificationPermissions;
#endif

#if PLATFORM(COCOA)
    RetainPtr<CFDataRef> networkATSContext;
#endif

#if PLATFORM(WAYLAND)
    String waylandCompositorDisplayName;
#endif

#if PLATFORM(COCOA)
    Vector<String> mediaMIMETypes;
#endif

#if PLATFORM(COCOA) || PLATFORM(GTK) || (PLATFORM(WPE) && ENABLE(WPE_PLATFORM))
    WebCore::ScreenProperties screenProperties;
#endif

#if !RELEASE_LOG_DISABLED
    bool shouldLogUserInteraction { false };
#endif

#if PLATFORM(MAC)
    bool useOverlayScrollbars { true };
#endif

#if USE(WPE_RENDERER)
    bool isServiceWorkerProcess { false };
    UnixFileDescriptor hostClientFileDescriptor;
    CString implementationLibraryName;
#endif

    std::optional<WebProcessDataStoreParameters> websiteDataStoreParameters;
    
#if PLATFORM(IOS) || PLATFORM(VISION)
    Vector<SandboxExtension::Handle> compilerServiceExtensionHandles;
#endif

    std::optional<SandboxExtension::Handle> mobileGestaltExtensionHandle;
    std::optional<SandboxExtension::Handle> launchServicesExtensionHandle;
#if HAVE(VIDEO_RESTRICTED_DECODING)
#if PLATFORM(MAC) || PLATFORM(MACCATALYST)
    SandboxExtension::Handle trustdExtensionHandle;
#endif
    bool enableDecodingHEIC { false };
    bool enableDecodingAVIF { false };
#endif

#if PLATFORM(IOS_FAMILY)
    Vector<SandboxExtension::Handle> dynamicIOKitExtensionHandles;
#endif

#if PLATFORM(VISION)
    // FIXME: Remove when GPU Process is fully enabled.
    Vector<SandboxExtension::Handle> metalCacheDirectoryExtensionHandles;
#endif

#if PLATFORM(COCOA)
    bool systemHasBattery { false };
    bool systemHasAC { false };
#endif

#if PLATFORM(IOS_FAMILY)
    PAL::UserInterfaceIdiom currentUserInterfaceIdiom { PAL::UserInterfaceIdiom::Default };
    bool supportsPictureInPicture { false };
    WebCore::RenderThemeIOS::CSSValueToSystemColorMap cssValueToSystemColorMap;
    WebCore::Color focusRingColor;
    String localizedDeviceModel;
    String contentSizeCategory;
#endif

#if USE(GBM)
    String renderDeviceFile;
#endif

#if PLATFORM(GTK) || PLATFORM(WPE)
    OptionSet<RendererBufferTransportMode> rendererBufferTransportMode;
    WebCore::SystemSettings::State systemSettings;
#endif

#if PLATFORM(GTK)
    bool useSystemAppearanceForScrollbars { false };
#endif

#if HAVE(CATALYST_USER_INTERFACE_IDIOM_AND_SCALE_FACTOR)
    std::pair<int64_t, double> overrideUserInterfaceIdiomAndScale;
#endif

#if HAVE(IOSURFACE)
    WebCore::IntSize maximumIOSurfaceSize;
    size_t bytesPerRowIOSurfaceAlignment;
#endif
    
    AccessibilityPreferences accessibilityPreferences;
#if PLATFORM(IOS_FAMILY)
    bool applicationAccessibilityEnabled { false };
#endif

#if PLATFORM(GTK) || PLATFORM(WPE)
    std::optional<MemoryPressureHandler::Configuration> memoryPressureHandlerConfiguration;
    bool disableFontHintingForTesting { false };
#endif

#if USE(GLIB)
    String applicationID;
    String applicationName;
#if ENABLE(REMOTE_INSPECTOR)
    CString inspectorServerAddress;
#endif
#endif

#if USE(ATSPI)
    String accessibilityBusAddress;
    String accessibilityBusName;
#endif

    String timeZoneOverride;

    HashMap<WebCore::RegistrableDomain, String> storageAccessUserAgentStringQuirksData;
    HashSet<WebCore::RegistrableDomain> storageAccessPromptQuirksDomains;
    ScriptTelemetryRules scriptTelemetryRules;

    Seconds memoryFootprintPollIntervalForTesting;
    Vector<size_t> memoryFootprintNotificationThresholds;

#if ENABLE(NOTIFY_BLOCKING)
    Vector<std::pair<String, uint64_t>> notifyState;
#endif
};

} // namespace WebKit
