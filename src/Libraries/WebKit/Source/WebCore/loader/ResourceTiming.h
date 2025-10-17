/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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

#include "LoaderMalloc.h"
#include "NetworkLoadMetrics.h"
#include "ResourceLoadTiming.h"
#include "ServerTiming.h"
#include <wtf/URL.h>

namespace WebCore {

class CachedResource;
class PerformanceServerTiming;
class ResourceResponse;
class ResourceLoadTiming;
class SecurityOrigin;

class ResourceTiming {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    static ResourceTiming fromMemoryCache(const URL&, const String& initiator, const ResourceLoadTiming&, const ResourceResponse&, const NetworkLoadMetrics&, const SecurityOrigin&);
    static ResourceTiming fromLoad(CachedResource&, const URL&, const String& initiator, const ResourceLoadTiming&, const NetworkLoadMetrics&, const SecurityOrigin&);
    static ResourceTiming fromSynchronousLoad(const URL&, const String& initiator, const ResourceLoadTiming&, const NetworkLoadMetrics&, const ResourceResponse&, const SecurityOrigin&);

    const URL& url() const { return m_url; }
    const String& initiatorType() const { return m_initiatorType; }
    const ResourceLoadTiming& resourceLoadTiming() const { return m_resourceLoadTiming; }
    const NetworkLoadMetrics& networkLoadMetrics() const { return m_networkLoadMetrics; }
    NetworkLoadMetrics& networkLoadMetrics() { return m_networkLoadMetrics; }
    Vector<Ref<PerformanceServerTiming>> populateServerTiming() const;
    bool isSameOriginRequest() const { return m_isSameOriginRequest; }
    ResourceTiming isolatedCopy() const &;
    ResourceTiming isolatedCopy() &&;

    void updateExposure(const SecurityOrigin&);
    void overrideInitiatorType(const String& type) { m_initiatorType = type; }
    bool isLoadedFromServiceWorker() const { return m_isLoadedFromServiceWorker; }

private:
    ResourceTiming(const URL&, const String& initiator, const ResourceLoadTiming&, const NetworkLoadMetrics&, const ResourceResponse&, const SecurityOrigin&);
    ResourceTiming(URL&& url, String&& initiatorType, const ResourceLoadTiming& resourceLoadTiming, NetworkLoadMetrics&& networkLoadMetrics, Vector<ServerTiming>&& serverTiming)
        : m_url(WTFMove(url))
        , m_initiatorType(WTFMove(initiatorType))
        , m_resourceLoadTiming(resourceLoadTiming)
        , m_networkLoadMetrics(WTFMove(networkLoadMetrics))
        , m_serverTiming(WTFMove(serverTiming))
    {
    }

    URL m_url;
    String m_initiatorType;
    ResourceLoadTiming m_resourceLoadTiming;
    NetworkLoadMetrics m_networkLoadMetrics;
    Vector<ServerTiming> m_serverTiming;
    bool m_isLoadedFromServiceWorker { false };
    bool m_isSameOriginRequest { false };
};

} // namespace WebCore
