/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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

#include <WebCore/Site.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakHashMap.h>
#include <wtf/WeakListHashSet.h>

namespace IPC {
class Connection;
}

namespace WebKit {

class FrameProcess;
class ProvisionalPageProxy;
class RemotePageProxy;
class WebPageProxy;
class WebPreferences;
class WebProcessProxy;

class BrowsingContextGroup : public RefCountedAndCanMakeWeakPtr<BrowsingContextGroup> {
public:
    static Ref<BrowsingContextGroup> create() { return adoptRef(*new BrowsingContextGroup()); }
    ~BrowsingContextGroup();

    Ref<FrameProcess> ensureProcessForSite(const WebCore::Site&, WebProcessProxy&, const WebPreferences&);
    Ref<FrameProcess> ensureProcessForConnection(IPC::Connection&, WebPageProxy&, const WebPreferences&);
    FrameProcess* processForSite(const WebCore::Site&);
    void addFrameProcess(FrameProcess&);
    void removeFrameProcess(FrameProcess&);
    void processDidTerminate(WebPageProxy&, WebProcessProxy&);

    void addPage(WebPageProxy&);
    void removePage(WebPageProxy&);
    void forEachRemotePage(const WebPageProxy&, Function<void(RemotePageProxy&)>&&);

    RemotePageProxy* remotePageInProcess(const WebPageProxy&, const WebProcessProxy&);

    RefPtr<RemotePageProxy> takeRemotePageInProcessForProvisionalPage(const WebPageProxy&, const WebProcessProxy&);
    void transitionPageToRemotePage(WebPageProxy&, const WebCore::Site& openerSite);
    void transitionProvisionalPageToRemotePage(ProvisionalPageProxy&, const WebCore::Site& provisionalNavigationFailureSite);

    bool hasRemotePages(const WebPageProxy&);

private:
    BrowsingContextGroup();

    HashMap<WebCore::Site, WeakPtr<FrameProcess>> m_processMap;
    WeakListHashSet<WebPageProxy> m_pages;
    WeakHashMap<WebPageProxy, HashSet<Ref<RemotePageProxy>>> m_remotePages;
};

}
