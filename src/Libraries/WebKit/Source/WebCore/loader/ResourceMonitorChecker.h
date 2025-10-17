/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 9, 2023.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include "ContentExtensionsBackend.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Lock.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/OptionSet.h>
#include <wtf/WorkQueue.h>

namespace WebCore {

class LocalFrame;

enum class ResourceMonitorEligibility : uint8_t { Unsure, NotEligible, Eligible };

class ResourceMonitorChecker final {
    friend MainThreadNeverDestroyed<ResourceMonitorChecker>;
public:
    using Eligibility = ResourceMonitorEligibility;

    WEBCORE_EXPORT static ResourceMonitorChecker& singleton();

    ~ResourceMonitorChecker();

    void checkEligibility(ContentExtensions::ResourceLoadInfo&&, CompletionHandler<void(Eligibility)>&&);
    bool checkNetworkUsageExceedingThreshold(size_t usage) const { return usage >= m_networkUsageThreshold; }

    WEBCORE_EXPORT void setContentRuleList(ContentExtensions::ContentExtensionsBackend&&);
    WEBCORE_EXPORT void setNetworkUsageThreshold(size_t threshold, double randomness = networkUsageThresholdRandomness);

    static constexpr Seconds ruleListPreparationTimeout = 10_s;
    static constexpr auto defaultEligibility = ResourceMonitorEligibility::NotEligible;
    WEBCORE_EXPORT static constexpr size_t networkUsageThreshold = 4 * MB;
    WEBCORE_EXPORT static constexpr double networkUsageThresholdRandomness = 0.0325;

private:
    ResourceMonitorChecker();

    Eligibility checkEligibility(const ContentExtensions::ResourceLoadInfo&);
    void finishPendingQueries(Function<Eligibility(const ContentExtensions::ResourceLoadInfo&)> checker);

    Ref<WorkQueue> protectedWorkQueue() { return m_workQueue; }

    Ref<WorkQueue> m_workQueue;
    std::unique_ptr<ContentExtensions::ContentExtensionsBackend> m_ruleList;
    Vector<std::pair<ContentExtensions::ResourceLoadInfo, CompletionHandler<void(Eligibility)>>> m_pendingQueries;
    bool m_ruleListIsPreparing { true };
    size_t m_networkUsageThreshold;
};

} // namespace WebCore

#endif
