/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 21, 2022.
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
#include "config.h"
#include "APIProcessPoolConfiguration.h"

#include "WebProcessPool.h"
#include "WebsiteDataStore.h"

namespace API {

Ref<ProcessPoolConfiguration> ProcessPoolConfiguration::create()
{
    return adoptRef(*new ProcessPoolConfiguration);
}

ProcessPoolConfiguration::ProcessPoolConfiguration() = default;

ProcessPoolConfiguration::~ProcessPoolConfiguration() = default;

Ref<ProcessPoolConfiguration> ProcessPoolConfiguration::copy()
{
    auto copy = this->create();

    copy->m_injectedBundlePath = this->m_injectedBundlePath;
    copy->m_cachePartitionedURLSchemes = this->m_cachePartitionedURLSchemes;
    copy->m_alwaysRevalidatedURLSchemes = this->m_alwaysRevalidatedURLSchemes;
    copy->m_additionalReadAccessAllowedPaths = this->m_additionalReadAccessAllowedPaths;
    copy->m_fullySynchronousModeIsAllowedForTesting = this->m_fullySynchronousModeIsAllowedForTesting;
    copy->m_ignoreSynchronousMessagingTimeoutsForTesting = this->m_ignoreSynchronousMessagingTimeoutsForTesting;
    copy->m_attrStyleEnabled = this->m_attrStyleEnabled;
    copy->m_shouldThrowExceptionForGlobalConstantRedeclaration = this->m_shouldThrowExceptionForGlobalConstantRedeclaration;
    copy->m_alwaysRunsAtBackgroundPriority = this->m_alwaysRunsAtBackgroundPriority;
    copy->m_shouldTakeUIBackgroundAssertion = this->m_shouldTakeUIBackgroundAssertion;
    copy->m_shouldCaptureDisplayInUIProcess = this->m_shouldCaptureDisplayInUIProcess;
    copy->m_shouldConfigureJSCForTesting = this->m_shouldConfigureJSCForTesting;
    copy->m_isJITEnabled = this->m_isJITEnabled;
    copy->m_presentingApplicationPID = this->m_presentingApplicationPID;
    copy->m_processSwapsOnNavigationFromClient = this->m_processSwapsOnNavigationFromClient;
    copy->m_processSwapsOnNavigationFromExperimentalFeatures = this->m_processSwapsOnNavigationFromExperimentalFeatures;
    copy->m_alwaysKeepAndReuseSwappedProcesses = this->m_alwaysKeepAndReuseSwappedProcesses;
    copy->m_processSwapsOnNavigationWithinSameNonHTTPFamilyProtocol = this->m_processSwapsOnNavigationWithinSameNonHTTPFamilyProtocol;
    copy->m_isAutomaticProcessWarmingEnabledByClient = this->m_isAutomaticProcessWarmingEnabledByClient;
    copy->m_usesWebProcessCache = this->m_usesWebProcessCache;
    copy->m_usesBackForwardCache = this->m_usesBackForwardCache;
    copy->m_usesSingleWebProcess = m_usesSingleWebProcess;
#if PLATFORM(GTK) && !USE(GTK4) && USE(CAIRO)
    copy->m_useSystemAppearanceForScrollbars = m_useSystemAppearanceForScrollbars;
#endif
#if PLATFORM(PLAYSTATION)
    copy->m_webProcessPath = this->m_webProcessPath;
    copy->m_networkProcessPath = this->m_networkProcessPath;
    copy->m_userId = this->m_userId;
#endif
#if PLATFORM(GTK) || PLATFORM(WPE)
    copy->m_memoryPressureHandlerConfiguration = this->m_memoryPressureHandlerConfiguration;
    copy->m_disableFontHintingForTesting = this->m_disableFontHintingForTesting;
#endif
#if HAVE(AUDIT_TOKEN)
    copy->m_presentingApplicationProcessToken = this->m_presentingApplicationProcessToken;
#endif
    copy->m_timeZoneOverride = this->m_timeZoneOverride;
    copy->m_memoryFootprintPollIntervalForTesting = this->m_memoryFootprintPollIntervalForTesting;
    copy->m_memoryFootprintNotificationThresholds = this->m_memoryFootprintNotificationThresholds;
#if ENABLE(WEB_PROCESS_SUSPENSION_DELAY)
    copy->m_suspendsWebProcessesAggressivelyOnMemoryPressure = this->m_suspendsWebProcessesAggressivelyOnMemoryPressure;
#endif
    return copy;
}

} // namespace API
