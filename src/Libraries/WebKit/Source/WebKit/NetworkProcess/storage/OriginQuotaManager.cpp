/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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
#include "OriginQuotaManager.h"

#include <wtf/SetForScope.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

static constexpr double defaultReportedQuotaIncreaseFactor = 2.0;

WTF_MAKE_TZONE_ALLOCATED_IMPL(OriginQuotaManager);

Ref<OriginQuotaManager> OriginQuotaManager::create(Parameters&& parameters, GetUsageFunction&& getUsageFunction)
{
    return adoptRef(*new OriginQuotaManager(WTFMove(parameters), WTFMove(getUsageFunction)));
}

OriginQuotaManager::OriginQuotaManager(Parameters&& parameters, GetUsageFunction&& getUsageFunction)
    : m_quota(parameters.quota)
    , m_standardReportedQuota(parameters.standardReportedQuota)
    , m_initialQuota(parameters.quota)
    , m_getUsageFunction(WTFMove(getUsageFunction))
    , m_increaseQuotaFunction(WTFMove(parameters.increaseQuotaFunction))
    , m_notifySpaceGrantedFunction(WTFMove(parameters.notifySpaceGrantedFunction))
{
    ASSERT(m_quota);
}

uint64_t OriginQuotaManager::usage()
{
    // Estimated usage that includes granted space.
    if (m_quotaCountdown)
        return m_quota - m_quotaCountdown;

    if (!m_usage)
        m_usage = m_getUsageFunction();

    return std::min(*m_usage, m_quota);
}

void OriginQuotaManager::requestSpace(uint64_t spaceRequested, RequestCallback&& callback)
{
    m_requests.append(OriginQuotaManager::Request { spaceRequested, WTFMove(callback), std::nullopt });
    handleRequests();
}

void OriginQuotaManager::handleRequests()
{
    if (m_currentRequest)
        return;

    SetForScope isHandlingRequests(m_isHandlingRequests, true);

    while (!m_currentRequest && !m_requests.isEmpty()) {
        m_currentRequest = m_requests.takeFirst();
        if (grantWithCurrentQuota(m_currentRequest->spaceRequested)) {
            m_currentRequest->callback(Decision::Grant);
            m_currentRequest = std::nullopt;
            continue;
        }

        if (!m_increaseQuotaFunction) {
            m_currentRequest->callback(Decision::Deny);
            m_currentRequest = std::nullopt;
            continue;
        }
    
        m_currentRequest->identifier = QuotaIncreaseRequestIdentifier::generate();
        m_increaseQuotaFunction(*m_currentRequest->identifier, m_quota, *m_usage, m_currentRequest->spaceRequested);
    }
}

bool OriginQuotaManager::grantWithCurrentQuota(uint64_t spaceRequested)
{
    if (grantFastPath(spaceRequested))
        return true;

    // When OriginQuotaManager is used for the first time, we want to make sure its initial quota is bigger than existing disk usage,
    // based on the assumption that the quota was increased to at least the disk usage under user's permission before.
    bool shouldUpdateQuotaBasedOnUsage = !m_usage;
    m_usage = m_getUsageFunction();
    if (shouldUpdateQuotaBasedOnUsage) {
        auto defaultQuotaStep = m_quota / 10;
        m_quota = std::max(m_quota, defaultQuotaStep * ((*m_usage / defaultQuotaStep) + 1));
    }
    m_quotaCountdown = *m_usage < m_quota ? m_quota - *m_usage : 0;

    return grantFastPath(spaceRequested);
}

void OriginQuotaManager::spaceGranted(uint64_t amount)
{
    if (m_notifySpaceGrantedFunction)
        m_notifySpaceGrantedFunction(amount);
}

bool OriginQuotaManager::grantFastPath(uint64_t spaceRequested)
{
    if (spaceRequested <= m_quotaCountdown) {
        m_quotaCountdown -= spaceRequested;
        spaceGranted(spaceRequested);
        return true;
    }

    return false;
}

void OriginQuotaManager::didIncreaseQuota(QuotaIncreaseRequestIdentifier identifier, std::optional<uint64_t> newQuota)
{
    if (!m_currentRequest || m_currentRequest->identifier != identifier)
        return;

    if (newQuota) {
        m_quota = *newQuota;
        // Recalculate m_quotaCountdown based on usage.
        m_quotaCountdown = 0;
    }

    auto decision = grantWithCurrentQuota(m_currentRequest->spaceRequested) ? Decision::Grant : Decision::Deny;
    m_currentRequest->callback(decision);
    m_currentRequest = std::nullopt;

    if (!m_isHandlingRequests)
        handleRequests();
}

void OriginQuotaManager::resetQuotaUpdatedBasedOnUsageForTesting()
{
    resetQuotaForTesting();
    m_usage = std::nullopt;
}

void OriginQuotaManager::resetQuotaForTesting()
{
    m_quota = m_initialQuota;
    m_quotaCountdown = 0;
}

void OriginQuotaManager::updateParametersForTesting(Parameters&& parameters)
{
    m_quota = parameters.quota;
    m_standardReportedQuota = parameters.standardReportedQuota;
    m_increaseQuotaFunction = WTFMove(parameters.increaseQuotaFunction);
    m_notifySpaceGrantedFunction = WTFMove(parameters.notifySpaceGrantedFunction);
    m_initialQuota = m_quota;
    m_quotaCountdown = 0;
    m_usage = std::nullopt;
}

uint64_t OriginQuotaManager::reportedQuota()
{
    if (!m_standardReportedQuota)
        return m_quota;

    // Standard reported quota is at least double existing usage.
    auto expectedUsage = usage() * defaultReportedQuotaIncreaseFactor;
    while (expectedUsage > m_standardReportedQuota && m_standardReportedQuota < m_quota)
        m_standardReportedQuota *= defaultReportedQuotaIncreaseFactor;

    return std::min(m_quota, m_standardReportedQuota);
}

} // namespace WebKit
