/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 9, 2025.
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

#include <wtf/text/WTFString.h>

namespace WebCore {

enum class HTMLMediaElementSourceType : uint8_t;

class DiagnosticLoggingKeys {
public:
    static String applicationCacheKey();
#if ENABLE(APPLICATION_MANIFEST)
    static String applicationManifestKey();
#endif
    static String audioKey();
    WEBCORE_EXPORT static String backNavigationDeltaKey();
    WEBCORE_EXPORT static String cacheControlNoStoreKey();
    static String cachedResourceRevalidationKey();
    static String cachedResourceRevalidationReasonKey();
    static String canCacheKey();
    WEBCORE_EXPORT static String canceledLessThan2SecondsKey();
    WEBCORE_EXPORT static String canceledLessThan5SecondsKey();
    WEBCORE_EXPORT static String canceledLessThan20SecondsKey();
    WEBCORE_EXPORT static String canceledMoreThan20SecondsKey();
    static String cannotSuspendActiveDOMObjectsKey();
    WEBCORE_EXPORT static String cpuUsageKey();
    WEBCORE_EXPORT static String createSharedBufferFailedKey();
    static String deniedByClientKey();
    static String deviceMotionKey();
    static String deviceOrientationKey();
    static String diskCacheKey();
    static String diskCacheAfterValidationKey();
    static String memoryCacheKey();
    static String memoryCacheAfterValidationKey();
    static String documentLoaderStoppingKey();
    WEBCORE_EXPORT static String domainCausingCrashKey();
    static String domainCausingEnergyDrainKey();
    WEBCORE_EXPORT static String domainCausingJetsamKey();
    WEBCORE_EXPORT static String simulatedPageCrashKey();
    WEBCORE_EXPORT static String exceededActiveMemoryLimitKey();
    WEBCORE_EXPORT static String exceededInactiveMemoryLimitKey();
    WEBCORE_EXPORT static String exceededBackgroundCPULimitKey();
    static String domainVisitedKey();
    static String engineFailedToLoadKey();
    WEBCORE_EXPORT static String entryRightlyNotWarmedUpKey();
    WEBCORE_EXPORT static String entryWronglyNotWarmedUpKey();
    static String expiredKey();
    WEBCORE_EXPORT static String failedLessThan2SecondsKey();
    WEBCORE_EXPORT static String failedLessThan5SecondsKey();
    WEBCORE_EXPORT static String failedLessThan20SecondsKey();
    WEBCORE_EXPORT static String failedMoreThan20SecondsKey();
    static String fontKey();
    static String httpsNoStoreKey();
    static String imageKey();
    WEBCORE_EXPORT static String internalErrorKey();
    WEBCORE_EXPORT static String invalidSessionIDKey();
    WEBCORE_EXPORT static String isAttachmentKey();
    WEBCORE_EXPORT static String isConditionalRequestKey();
    static String isDisabledKey();
    static String isErrorPageKey();
    WEBCORE_EXPORT static String isReloadIgnoringCacheDataKey();
    static String loadingKey();
    static String isLoadingKey();
    static String mainResourceKey();
    static String mediaLoadedKey();
    static String mediaLoadingFailedKey();
    static String memoryCacheEntryDecisionKey();
    static String memoryCacheUsageKey();
    WEBCORE_EXPORT static String missingValidatorFieldsKey();
    static String navigationKey();
    WEBCORE_EXPORT static String needsRevalidationKey();
    WEBCORE_EXPORT static String networkCacheKey();
    WEBCORE_EXPORT static String networkCacheFailureReasonKey();
    WEBCORE_EXPORT static String networkCacheUnusedReasonKey();
    WEBCORE_EXPORT static String networkCacheReuseFailureKey();
    static String networkKey();
    WEBCORE_EXPORT static String networkProcessCrashedKey();
    WEBCORE_EXPORT static String neverSeenBeforeKey();
    static String noKey();
    static String noCurrentHistoryItemKey();
    static String noDocumentLoaderKey();
    WEBCORE_EXPORT static String noLongerInCacheKey();
    WEBCORE_EXPORT static String nonVisibleStateKey();
    WEBCORE_EXPORT static String notHTTPFamilyKey();
    WEBCORE_EXPORT static String occurredKey();
    WEBCORE_EXPORT static String otherKey();
    static String backForwardCacheKey();
    static String backForwardCacheFailureKey();
    static String visuallyEmptyKey();
    static String pageContainsAtLeastOneMediaEngineKey();
    static String pageContainsMediaEngineKey();
    static String pageLoadedKey();
    static String playedKey();
    static String postPageBackgroundingCPUUsageKey();
    static String postPageBackgroundingMemoryUsageKey();
    static String postPageLoadCPUUsageKey();
    static String postPageLoadMemoryUsageKey();
    static String provisionalLoadKey();
    static String prunedDueToMaxSizeReached();
    static String prunedDueToMemoryPressureKey();
    static String prunedDueToProcessSuspended();
    static String quirkRedirectComingKey();
    static String rawKey();
    static String redirectKey();
    static String reloadFromOriginKey();
    static String reloadKey();
    static String reloadRevalidatingExpiredKey();
    static String replaceKey();
    static String resourceLoadedKey();
    static String resourceResponseSourceKey();
    WEBCORE_EXPORT static String retrievalKey();
    WEBCORE_EXPORT static String retrievalRequestKey();
    static String sameLoadKey();
    static String scriptKey();
    static String serviceWorkerKey();
    static String siteSpecificQuirkKey();
    WEBCORE_EXPORT static String streamingMedia();
    static String styleSheetKey();
    WEBCORE_EXPORT static String succeededLessThan2SecondsKey();
    WEBCORE_EXPORT static String succeededLessThan5SecondsKey();
    WEBCORE_EXPORT static String succeededLessThan20SecondsKey();
    WEBCORE_EXPORT static String succeededMoreThan20SecondsKey();
    WEBCORE_EXPORT static String successfulSpeculativeWarmupWithRevalidationKey();
    WEBCORE_EXPORT static String successfulSpeculativeWarmupWithoutRevalidationKey();
    static String svgDocumentKey();
    WEBCORE_EXPORT static String synchronousMessageFailedKey();
    WEBCORE_EXPORT static String telemetryPageLoadKey();
    WEBCORE_EXPORT static String timedOutKey();
    WEBCORE_EXPORT static String uncacheableStatusCodeKey();
    static String underMemoryPressureKey();
    WEBCORE_EXPORT static String unknownEntryRequestKey();
    WEBCORE_EXPORT static String unlikelyToReuseKey();
    WEBCORE_EXPORT static String unsupportedHTTPMethodKey();
    static String unsuspendableDOMObjectKey();
    WEBCORE_EXPORT static String unusedKey();
    static String usedKey();
    WEBCORE_EXPORT static String userZoomActionKey();
    WEBCORE_EXPORT static String varyingHeaderMismatchKey();
    static String videoKey();
    WEBCORE_EXPORT static String visibleNonActiveStateKey();
    WEBCORE_EXPORT static String visibleAndActiveStateKey();
    WEBCORE_EXPORT static String wastedSpeculativeWarmupWithRevalidationKey();
    WEBCORE_EXPORT static String wastedSpeculativeWarmupWithoutRevalidationKey();
    WEBCORE_EXPORT static String webViewKey();
    static String yesKey();

    static String mediaSourceTypeWatchTimeKey();
    static String mediaVideoCodecWatchTimeKey();
    static String mediaAudioCodecWatchTimeKey();
    static String mediaBufferingWatchTimeKey();
    static String mediaTextTrackWatchTimeKey();

    static String secondsKey();
    static String sourceTypeKey();
    static String videoCodecKey();
    static String audioCodecKey();
    static String textTrackTypeKey();
    static String textTrackKindKey();
    static String textTrackModeKey();

    WEBCORE_EXPORT static String memoryUsageToDiagnosticLoggingKey(uint64_t memoryUsage);
    WEBCORE_EXPORT static String foregroundCPUUsageToDiagnosticLoggingKey(double cpuUsage);
    WEBCORE_EXPORT static String backgroundCPUUsageToDiagnosticLoggingKey(double cpuUsage);
};

} // namespace WebCore
