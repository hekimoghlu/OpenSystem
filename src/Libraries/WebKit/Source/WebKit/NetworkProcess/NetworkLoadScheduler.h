/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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

#include <WebCore/LoadSchedulingMode.h>
#include <WebCore/NetworkLoadMetrics.h>
#include <WebCore/PageIdentifier.h>
#include <tuple>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakListHashSet.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class ResourceError;
}

namespace WebKit {

class NetworkLoad;

class NetworkLoadScheduler : public RefCountedAndCanMakeWeakPtr<NetworkLoadScheduler> {
    WTF_MAKE_TZONE_ALLOCATED(NetworkLoadScheduler);
public:
    static Ref<NetworkLoadScheduler> create()
    {
        return adoptRef(*new NetworkLoadScheduler);
    }

    ~NetworkLoadScheduler();

    void schedule(NetworkLoad&);
    void unschedule(NetworkLoad&, const WebCore::NetworkLoadMetrics* = nullptr);

    void startedPreconnectForMainResource(const URL&, const String& userAgent);
    void finishedPreconnectForMainResource(const URL&, const String& userAgent, const WebCore::ResourceError&);

    void setResourceLoadSchedulingMode(WebCore::PageIdentifier, WebCore::LoadSchedulingMode);
    void prioritizeLoads(const Vector<NetworkLoad*>&);
    void clearPageData(WebCore::PageIdentifier);

private:
    NetworkLoadScheduler();

    void scheduleLoad(NetworkLoad&);
    void unscheduleLoad(NetworkLoad&);

    void scheduleMainResourceLoad(NetworkLoad&);
    void unscheduleMainResourceLoad(NetworkLoad&, const WebCore::NetworkLoadMetrics*);

    bool isOriginHTTP1X(const String&);
    void updateOriginProtocolInfo(const String&, const String&);

    class HostContext;
    HostContext* contextForLoad(const NetworkLoad&);

    using PageContext = HashMap<String, std::unique_ptr<HostContext>>;
    HashMap<WebCore::PageIdentifier, std::unique_ptr<PageContext>> m_pageContexts;

    struct PendingMainResourcePreconnectInfo {
        unsigned pendingPreconnects {1};
        WeakListHashSet<NetworkLoad> pendingLoads;
    };
    // Maps (protocolHostAndPort, userAgent) => PendingMainResourcePreconnectInfo.
    using PendingPreconnectMap = HashMap<std::tuple<String, String>, PendingMainResourcePreconnectInfo>;
    PendingPreconnectMap m_pendingMainResourcePreconnects;

    void maybePrunePreconnectInfo(PendingPreconnectMap::iterator&);

    HashSet<String> m_http1XOrigins;
};

}
