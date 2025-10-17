/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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

#include "QuotaIncreaseRequestIdentifier.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Deque.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace WebKit {

class OriginQuotaManager : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<OriginQuotaManager> {
    WTF_MAKE_TZONE_ALLOCATED(OriginQuotaManager);
public:
    using GetUsageFunction = Function<uint64_t()>;
    using IncreaseQuotaFunction = Function<void(QuotaIncreaseRequestIdentifier, uint64_t currentQuota, uint64_t currentUsage, uint64_t requestedIncrease)>;
    using NotifySpaceGrantedFunction = Function<void(uint64_t)>;
    struct Parameters {
        uint64_t quota { 0 };
        uint64_t standardReportedQuota { 0 };
        IncreaseQuotaFunction increaseQuotaFunction;
        NotifySpaceGrantedFunction notifySpaceGrantedFunction;
    };
    static Ref<OriginQuotaManager> create(Parameters&&, GetUsageFunction&&);
    uint64_t reportedQuota();
    uint64_t usage();
    enum class Decision : bool { Deny, Grant };
    using RequestCallback = CompletionHandler<void(Decision)>;
    void requestSpace(uint64_t spaceRequested, RequestCallback&&);
    void didIncreaseQuota(QuotaIncreaseRequestIdentifier, std::optional<uint64_t> newQuota);

    void resetQuotaUpdatedBasedOnUsageForTesting();
    void resetQuotaForTesting();
    void updateParametersForTesting(Parameters&&);

private:
    OriginQuotaManager(Parameters&&, GetUsageFunction&&);
    void handleRequests();
    bool grantWithCurrentQuota(uint64_t spaceRequested);
    bool grantFastPath(uint64_t spaceRequested);
    void spaceGranted(uint64_t amount);

    struct Request {
        uint64_t spaceRequested;
        RequestCallback callback;
        Markable<QuotaIncreaseRequestIdentifier> identifier;
    };
    Deque<Request> m_requests;
    std::optional<Request> m_currentRequest;
    bool m_isHandlingRequests { false };
    uint64_t m_quotaCountdown { 0 };
    uint64_t m_quota;
    uint64_t m_standardReportedQuota;
    uint64_t m_initialQuota; // Test only.
    std::optional<uint64_t> m_usage;
    GetUsageFunction m_getUsageFunction;
    IncreaseQuotaFunction m_increaseQuotaFunction;
    NotifySpaceGrantedFunction m_notifySpaceGrantedFunction;
};

} // namespace WebKit
