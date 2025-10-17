/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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
#ifndef WKContext_h
#define WKContext_h

#include <WebKit/WKBase.h>
#include <WebKit/WKContextDownloadClient.h>
#include <WebKit/WKContextHistoryClient.h>
#include <WebKit/WKContextInjectedBundleClient.h>
#include <WebKit/WKDeprecated.h>
#include <WebKit/WKProcessTerminationReason.h>

#if defined(WIN32) || defined(_WIN32)
typedef int WKProcessID;
#else
#include <unistd.h>
typedef pid_t WKProcessID;
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
    kWKCacheModelDocumentViewer = 0,
    kWKCacheModelDocumentBrowser = 1,
    kWKCacheModelPrimaryWebBrowser = 2
};
typedef uint32_t WKCacheModel;

// Context Client
typedef void (*WKContextPlugInAutoStartOriginHashesChangedCallback)(WKContextRef context, const void *clientInfo);
typedef void (*WKContextPlugInInformationBecameAvailableCallback)(WKContextRef context, WKArrayRef plugIn, const void *clientInfo);
typedef WKDataRef (*WKContextCopyWebCryptoMasterKeyCallback)(WKContextRef context, const void *clientInfo);

typedef void (*WKContextChildProcessDidCrashCallback)(WKContextRef context, const void *clientInfo);
typedef WKContextChildProcessDidCrashCallback WKContextNetworkProcessDidCrashCallback;

typedef void (*WKContextChildProcessWithPIDDidCrashCallback)(WKContextRef context, WKProcessID processID, const void *clientInfo);
typedef void (*WKContextChildProcessDidCrashWithDetailsCallback)(WKContextRef context, WKProcessID processID, WKProcessTerminationReason reason, const void *clientInfo);

typedef struct WKContextClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKContextClientBase;

typedef struct WKContextClientV0 {
    WKContextClientBase                                                 base;

    // Version 0.
    WKContextPlugInAutoStartOriginHashesChangedCallback                 plugInAutoStartOriginHashesChanged;
    WKContextNetworkProcessDidCrashCallback                             networkProcessDidCrash;
    WKContextPlugInInformationBecameAvailableCallback                   plugInInformationBecameAvailable;
} WKContextClientV0;

typedef struct WKContextClientV1 {
    WKContextClientBase                                                 base;

    // Version 0.
    WKContextPlugInAutoStartOriginHashesChangedCallback                 plugInAutoStartOriginHashesChanged;
    WKContextNetworkProcessDidCrashCallback                             networkProcessDidCrash;
    WKContextPlugInInformationBecameAvailableCallback                   plugInInformationBecameAvailable;

    // Version 1.
    void                                                                (*copyWebCryptoMasterKey_unavailable)(void);
} WKContextClientV1;

// WKContextClientV1 and WKContextClientV2 are identical.
typedef struct WKContextClientV2 {
    WKContextClientBase                                                 base;

    // Version 0.
    WKContextPlugInAutoStartOriginHashesChangedCallback                 plugInAutoStartOriginHashesChanged;
    WKContextNetworkProcessDidCrashCallback                             networkProcessDidCrash;
    WKContextPlugInInformationBecameAvailableCallback                   plugInInformationBecameAvailable;

    // Version 1.
    void                                                                (*copyWebCryptoMasterKey_unavailable)(void);
} WKContextClientV2;

typedef struct WKContextClientV3 {
    WKContextClientBase                                                 base;

    // Version 0.
    WKContextPlugInAutoStartOriginHashesChangedCallback                 plugInAutoStartOriginHashesChanged;
    WKContextNetworkProcessDidCrashCallback                             networkProcessDidCrash;
    WKContextPlugInInformationBecameAvailableCallback                   plugInInformationBecameAvailable;

    // Version 1.
    void                                                                (*copyWebCryptoMasterKey_unavailable)(void);

    // Version 3.
    WKContextChildProcessWithPIDDidCrashCallback                        serviceWorkerProcessDidCrash;
    WKContextChildProcessWithPIDDidCrashCallback                        gpuProcessDidCrash;
} WKContextClientV3;

typedef struct WKContextClientV4 {
    WKContextClientBase                                                 base;

    // Version 0.
    WKContextPlugInAutoStartOriginHashesChangedCallback                 plugInAutoStartOriginHashesChanged;
    WKContextNetworkProcessDidCrashCallback                             networkProcessDidCrash;
    WKContextPlugInInformationBecameAvailableCallback                   plugInInformationBecameAvailable;

    // Version 1.
    void                                                                (*copyWebCryptoMasterKey_unavailable)(void);

    // Version 3.
    WKContextChildProcessWithPIDDidCrashCallback                        serviceWorkerProcessDidCrash;
    WKContextChildProcessWithPIDDidCrashCallback                        gpuProcessDidCrash;
    
    // Version 4.
    WKContextChildProcessDidCrashWithDetailsCallback                    networkProcessDidCrashWithDetails;
    WKContextChildProcessDidCrashWithDetailsCallback                    serviceWorkerProcessDidCrashWithDetails;
    WKContextChildProcessDidCrashWithDetailsCallback                    gpuProcessDidCrashWithDetails;
} WKContextClientV4;


// FIXME: Remove these once support for Mavericks has been dropped.
enum {
    kWKProcessModelSharedSecondaryProcess = 0,
    kWKProcessModelMultipleSecondaryProcesses = 1
};
typedef uint32_t WKProcessModel;

enum {
    kWKStatisticsOptionsWebContent = 1 << 0,
    kWKStatisticsOptionsNetworking = 1 << 1
};
typedef uint32_t WKStatisticsOptions;

WK_EXPORT WKTypeID WKContextGetTypeID(void);

WK_EXPORT WKContextRef WKContextCreate(void) WK_C_API_DEPRECATED_WITH_REPLACEMENT(WKContextCreateWithConfiguration);
WK_EXPORT WKContextRef WKContextCreateWithInjectedBundlePath(WKStringRef path) WK_C_API_DEPRECATED_WITH_REPLACEMENT(WKContextCreateWithConfiguration);
WK_EXPORT WKContextRef WKContextCreateWithConfiguration(WKContextConfigurationRef configuration);

WK_EXPORT void WKContextSetClient(WKContextRef context, const WKContextClientBase* client);
WK_EXPORT void WKContextSetInjectedBundleClient(WKContextRef context, const WKContextInjectedBundleClientBase* client);
WK_EXPORT void WKContextSetHistoryClient(WKContextRef context, const WKContextHistoryClientBase* client);
WK_EXPORT void WKContextSetDownloadClient(WKContextRef context, const WKContextDownloadClientBase* client) WK_C_API_DEPRECATED_WITH_REPLACEMENT(WKDownload);

WK_EXPORT WKDownloadRef WKContextDownloadURLRequest(WKContextRef context, WKURLRequestRef request) WK_C_API_DEPRECATED;
WK_EXPORT WKDownloadRef WKContextResumeDownload(WKContextRef context, WKDataRef resumeData, WKStringRef path) WK_C_API_DEPRECATED;

WK_EXPORT void WKContextSetInitializationUserDataForInjectedBundle(WKContextRef context, WKTypeRef userData);
WK_EXPORT void WKContextPostMessageToInjectedBundle(WKContextRef context, WKStringRef messageName, WKTypeRef messageBody);

WK_EXPORT void WKContextAddVisitedLink(WKContextRef context, WKStringRef visitedURL);
WK_EXPORT void WKContextClearVisitedLinks(WKContextRef contextRef);

WK_EXPORT void WKContextSetCacheModel(WKContextRef context, WKCacheModel cacheModel);
WK_EXPORT WKCacheModel WKContextGetCacheModel(WKContextRef context);

// FIXME: Move these to WKDeprecatedFunctions.cpp once support for Mavericks has been dropped.
WK_EXPORT void WKContextSetProcessModel(WKContextRef, WKProcessModel);

WK_EXPORT void WKContextSetMaximumNumberOfProcesses(WKContextRef context, unsigned numberOfProcesses) WK_C_API_DEPRECATED;
WK_EXPORT unsigned WKContextGetMaximumNumberOfProcesses(WKContextRef context) WK_C_API_DEPRECATED;

WK_EXPORT void WKContextSetUsesSingleWebProcess(WKContextRef, bool);
WK_EXPORT bool WKContextGetUsesSingleWebProcess(WKContextRef);

WK_EXPORT void WKContextStartMemorySampler(WKContextRef context, WKDoubleRef interval);
WK_EXPORT void WKContextStopMemorySampler(WKContextRef context);

WK_EXPORT WKWebsiteDataStoreRef WKContextGetWebsiteDataStore(WKContextRef context) WK_C_API_DEPRECATED_WITH_REPLACEMENT(WKWebsiteDataStoreGetDefaultDataStore);

WK_EXPORT WKApplicationCacheManagerRef WKContextGetApplicationCacheManager(WKContextRef context) WK_C_API_DEPRECATED_WITH_REPLACEMENT(WKWebsiteDataStoreGetDefaultDataStore);
WK_EXPORT WKGeolocationManagerRef WKContextGetGeolocationManager(WKContextRef context);
WK_EXPORT WKIconDatabaseRef WKContextGetIconDatabase(WKContextRef context);
WK_EXPORT WKKeyValueStorageManagerRef WKContextGetKeyValueStorageManager(WKContextRef context) WK_C_API_DEPRECATED_WITH_REPLACEMENT(WKWebsiteDataStoreGetDefaultDataStore);
WK_EXPORT WKNotificationManagerRef WKContextGetNotificationManager(WKContextRef context);
WK_EXPORT WKResourceCacheManagerRef WKContextGetResourceCacheManager(WKContextRef context) WK_C_API_DEPRECATED_WITH_REPLACEMENT(WKWebsiteDataStoreGetDefaultDataStore);

typedef void (*WKContextGetStatisticsFunction)(WKDictionaryRef statistics, WKErrorRef error, void* functionContext);
WK_EXPORT void WKContextGetStatistics(WKContextRef context, void* functionContext, WKContextGetStatisticsFunction function);
WK_EXPORT void WKContextGetStatisticsWithOptions(WKContextRef context, WKStatisticsOptions statisticsMask, void* functionContext, WKContextGetStatisticsFunction function);

WK_EXPORT bool WKContextJavaScriptConfigurationFileEnabled(WKContextRef context);
WK_EXPORT void WKContextSetJavaScriptConfigurationFileEnabled(WKContextRef context, bool enable);
WK_EXPORT void WKContextGarbageCollectJavaScriptObjects(WKContextRef context);
WK_EXPORT void WKContextSetJavaScriptGarbageCollectorTimerEnabled(WKContextRef context, bool enable);

WK_EXPORT WKDictionaryRef WKContextCopyPlugInAutoStartOriginHashes(WKContextRef context);
WK_EXPORT void WKContextSetPlugInAutoStartOriginHashes(WKContextRef context, WKDictionaryRef dictionary);
WK_EXPORT void WKContextSetPlugInAutoStartOrigins(WKContextRef contextRef, WKArrayRef arrayRef);
WK_EXPORT void WKContextSetPlugInAutoStartOriginsFilteringOutEntriesAddedAfterTime(WKContextRef contextRef, WKDictionaryRef dictionaryRef, double time);
WK_EXPORT void WKContextRefreshPlugIns(WKContextRef context);

WK_EXPORT void WKContextSetCustomWebContentServiceBundleIdentifier(WKContextRef contextRef, WKStringRef name) WK_C_API_DEPRECATED;

WK_EXPORT void WKContextClearMockGamepadsForTesting(WKContextRef contextRef);

typedef void (*WKContextSetResourceMonitorURLsFunction)(void* functionContext);
WK_EXPORT void WKContextSetResourceMonitorURLsForTesting(WKContextRef contextRef, WKStringRef rulesText, void* context, WKContextSetResourceMonitorURLsFunction callback);

#ifdef __cplusplus
}
#endif

#endif /* WKContext_h */
