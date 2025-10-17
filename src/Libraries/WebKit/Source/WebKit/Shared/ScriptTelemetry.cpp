/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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
#include "ScriptTelemetry.h"

#include <WebCore/RegistrableDomain.h>
#include <WebCore/SecurityOrigin.h>

namespace WebKit {

static void initializeFilterRules(Vector<String>&& source, MemoryCompactRobinHoodHashSet<String>& target)
{
    target.reserveInitialCapacity(source.size());
    for (auto& host : source)
        target.add(host);
}

ScriptTelemetryFilter::ScriptTelemetryFilter(ScriptTelemetryRules&& rules)
{
    initializeFilterRules(WTFMove(rules.thirdPartyHosts), m_thirdPartyHosts);
    initializeFilterRules(WTFMove(rules.thirdPartyTopDomains), m_thirdPartyTopDomains);
    initializeFilterRules(WTFMove(rules.firstPartyHosts), m_firstPartyHosts);
    initializeFilterRules(WTFMove(rules.firstPartyTopDomains), m_firstPartyTopDomains);
}

bool ScriptTelemetryFilter::matches(const URL& url, const WebCore::SecurityOrigin& topOrigin)
{
    WebCore::RegistrableDomain scriptTopDomain { url };

    auto scriptTopDomainName = scriptTopDomain.string();
    if (scriptTopDomainName.isEmpty())
        return false;

    auto hostName = url.host().toStringWithoutCopying();
    if (hostName.isEmpty())
        return false;

    if (!scriptTopDomain.matches(topOrigin.data())) {
        if (m_thirdPartyHosts.contains(hostName))
            return true;

        if (m_thirdPartyTopDomains.contains(scriptTopDomainName))
            return true;
    }

    if (UNLIKELY(m_firstPartyHosts.contains(hostName)))
        return true;

    if (UNLIKELY(m_firstPartyTopDomains.contains(scriptTopDomainName)))
        return true;

    return false;
}

} // namespace WebKit
