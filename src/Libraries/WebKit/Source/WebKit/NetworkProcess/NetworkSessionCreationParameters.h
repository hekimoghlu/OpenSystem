/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 28, 2024.
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

#include "ResourceLoadStatisticsParameters.h"
#include "UnifiedOriginStorageLevel.h"
#include "WebPushDaemonConnectionConfiguration.h"
#include <WebCore/NetworkStorageSession.h>
#include <pal/SessionID.h>
#include <wtf/Seconds.h>
#include <wtf/URL.h>
#include <wtf/UUID.h>

#if USE(SOUP)
#include "SoupCookiePersistentStorageType.h"
#include <WebCore/HTTPCookieAcceptPolicy.h>
#include <WebCore/SoupNetworkProxySettings.h>
#endif

#if USE(CURL)
#include <WebCore/CurlProxySettings.h>
#endif

namespace WebKit {

enum class AllowsCellularAccess : bool { No, Yes };

struct NetworkSessionCreationParameters {
    PAL::SessionID sessionID { PAL::SessionID::defaultSessionID() };
    Markable<WTF::UUID> dataStoreIdentifier;
    String boundInterfaceIdentifier;
    AllowsCellularAccess allowsCellularAccess { AllowsCellularAccess::Yes };
#if PLATFORM(COCOA)
    RetainPtr<CFDictionaryRef> proxyConfiguration;
    String sourceApplicationBundleIdentifier;
    String sourceApplicationSecondaryIdentifier;
    bool shouldLogCookieInformation { false };
    URL httpProxy;
    URL httpsProxy;
#endif
#if HAVE(ALTERNATIVE_SERVICE)
    String alternativeServiceDirectory;
    SandboxExtension::Handle alternativeServiceDirectoryExtensionHandle;
#endif
    String hstsStorageDirectory;
    SandboxExtension::Handle hstsStorageDirectoryExtensionHandle;
#if USE(SOUP)
    String cookiePersistentStoragePath;
    SoupCookiePersistentStorageType cookiePersistentStorageType { SoupCookiePersistentStorageType::Text };
    bool persistentCredentialStorageEnabled { true };
    bool ignoreTLSErrors { false };
    WebCore::SoupNetworkProxySettings proxySettings;
    WebCore::HTTPCookieAcceptPolicy cookieAcceptPolicy { WebCore::HTTPCookieAcceptPolicy::ExclusivelyFromMainDocumentDomain };
#endif
#if USE(CURL)
    String cookiePersistentStorageFile;
    WebCore::CurlProxySettings proxySettings;
#endif
    bool deviceManagementRestrictionsEnabled { false };
    bool allLoadsBlockedByDeviceManagementRestrictionsForTesting { false };
    WebPushD::WebPushDaemonConnectionConfiguration webPushDaemonConnectionConfiguration;

    String networkCacheDirectory;
    SandboxExtension::Handle networkCacheDirectoryExtensionHandle;
    String dataConnectionServiceType;
    bool fastServerTrustEvaluationEnabled { false };
    bool networkCacheSpeculativeValidationEnabled { false };
    bool shouldUseTestingNetworkSession { false };
    bool staleWhileRevalidateEnabled { false };
    unsigned testSpeedMultiplier { 1 };
    bool suppressesConnectionTerminationOnSystemChange { false };
    bool allowsServerPreconnect { true };
    bool requiresSecureHTTPSProxyConnection { false };
    bool shouldRunServiceWorkersOnMainThreadForTesting { false };
    std::optional<unsigned> overrideServiceWorkerRegistrationCountTestingValue;
    bool preventsSystemHTTPProxyAuthentication { false };
    std::optional<bool> useNetworkLoader { std::nullopt };
    bool allowsHSTSWithUntrustedRootCertificate { false };
    String pcmMachServiceName;
    String webPushMachServiceName;
    String webPushPartitionString;
    bool enablePrivateClickMeasurementDebugMode { false };
    bool isBlobRegistryTopOriginPartitioningEnabled { false };
    bool isOptInCookiePartitioningEnabled { false };
    bool shouldSendPrivateTokenIPCForTesting { false };

    UnifiedOriginStorageLevel unifiedOriginStorageLevel { UnifiedOriginStorageLevel::Standard };
    uint64_t perOriginStorageQuota;
    std::optional<double> originQuotaRatio;
    std::optional<double> totalQuotaRatio;
    std::optional<uint64_t> standardVolumeCapacity;
    std::optional<uint64_t> volumeCapacityOverride;
    String localStorageDirectory;
    SandboxExtension::Handle localStorageDirectoryExtensionHandle;
    String indexedDBDirectory;
    SandboxExtension::Handle indexedDBDirectoryExtensionHandle;
    String cacheStorageDirectory;
    SandboxExtension::Handle cacheStorageDirectoryExtensionHandle;
    String generalStorageDirectory;
    SandboxExtension::Handle generalStorageDirectoryHandle;
    String serviceWorkerRegistrationDirectory;
    SandboxExtension::Handle serviceWorkerRegistrationDirectoryExtensionHandle;
    bool serviceWorkerProcessTerminationDelayEnabled { true };
    bool inspectionForServiceWorkersAllowed { true };
    bool storageSiteValidationEnabled { false };
#if ENABLE(DECLARATIVE_WEB_PUSH)
    bool isDeclarativeWebPushEnabled { false };
#endif
#if HAVE(NW_PROXY_CONFIG)
    std::optional<Vector<std::pair<Vector<uint8_t>, WTF::UUID>>> proxyConfigData;
#endif
    ResourceLoadStatisticsParameters resourceLoadStatisticsParameters;
};

} // namespace WebKit
