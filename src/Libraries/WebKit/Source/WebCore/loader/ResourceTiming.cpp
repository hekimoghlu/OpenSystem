/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 28, 2025.
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
#include "ResourceTiming.h"

#include "CachedResource.h"
#include "DocumentLoadTiming.h"
#include "OriginAccessPatterns.h"
#include "PerformanceServerTiming.h"
#include "SecurityOrigin.h"
#include "ServerTimingParser.h"
#include <wtf/CrossThreadCopier.h>

namespace WebCore {

ResourceTiming ResourceTiming::fromMemoryCache(const URL& url, const String& initiator, const ResourceLoadTiming& loadTiming, const ResourceResponse& response, const NetworkLoadMetrics& networkLoadMetrics, const SecurityOrigin& securityOrigin)
{
    return ResourceTiming(url, initiator, loadTiming, networkLoadMetrics, response, securityOrigin);
}

ResourceTiming ResourceTiming::fromLoad(CachedResource& resource, const URL& url, const String& initiator, const ResourceLoadTiming& loadTiming, const NetworkLoadMetrics& networkLoadMetrics, const SecurityOrigin& securityOrigin)
{
    return ResourceTiming(url, initiator, loadTiming, networkLoadMetrics, resource.response(), securityOrigin);
}

ResourceTiming ResourceTiming::fromSynchronousLoad(const URL& url, const String& initiator, const ResourceLoadTiming& loadTiming, const NetworkLoadMetrics& networkLoadMetrics, const ResourceResponse& response, const SecurityOrigin& securityOrigin)
{
    return ResourceTiming(url, initiator, loadTiming, networkLoadMetrics, response, securityOrigin);
}

ResourceTiming::ResourceTiming(const URL& url, const String& initiatorType, const ResourceLoadTiming& timing, const NetworkLoadMetrics& networkLoadMetrics, const ResourceResponse& response, const SecurityOrigin& origin)
    : m_url(url)
    , m_initiatorType(initiatorType)
    , m_resourceLoadTiming(timing)
    , m_networkLoadMetrics(networkLoadMetrics)
    , m_serverTiming(ServerTimingParser::parseServerTiming(response.httpHeaderField(HTTPHeaderName::ServerTiming)))
    , m_isLoadedFromServiceWorker(response.source() == ResourceResponse::Source::ServiceWorker)
    , m_isSameOriginRequest(!m_networkLoadMetrics.hasCrossOriginRedirect
        && origin.protocol() == url.protocol()
        && origin.host() == url.host()
        && origin.port() == url.port())
{
}

void ResourceTiming::updateExposure(const SecurityOrigin& origin)
{
    m_isSameOriginRequest = m_isSameOriginRequest && origin.canRequest(m_url, OriginAccessPatternsForWebProcess::singleton());
}

Vector<Ref<PerformanceServerTiming>> ResourceTiming::populateServerTiming() const
{
    // To increase privacy, this additional check was proposed at https://github.com/w3c/resource-timing/issues/342 .
    if (!m_isSameOriginRequest)
        return { };

    return WTF::map(m_serverTiming, [] (auto& entry) {
        return PerformanceServerTiming::create(String(entry.name), entry.duration, String(entry.description));
    });
}

ResourceTiming ResourceTiming::isolatedCopy() const &
{
    return ResourceTiming {
        m_url.isolatedCopy(),
        m_initiatorType.isolatedCopy(),
        m_resourceLoadTiming.isolatedCopy(),
        m_networkLoadMetrics.isolatedCopy(),
        crossThreadCopy(m_serverTiming)
    };
}

ResourceTiming ResourceTiming::isolatedCopy() &&
{
    return ResourceTiming {
        WTFMove(m_url).isolatedCopy(),
        WTFMove(m_initiatorType).isolatedCopy(),
        WTFMove(m_resourceLoadTiming).isolatedCopy(),
        WTFMove(m_networkLoadMetrics).isolatedCopy(),
        crossThreadCopy(WTFMove(m_serverTiming))
    };
}

} // namespace WebCore
