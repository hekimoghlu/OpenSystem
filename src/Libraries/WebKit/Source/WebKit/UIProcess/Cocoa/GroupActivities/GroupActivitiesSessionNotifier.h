/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 29, 2022.
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

#if ENABLE(MEDIA_SESSION_COORDINATOR) && HAVE(GROUP_ACTIVITIES)

#include "GroupActivitiesSession.h"
#include <wtf/HashMap.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URLHash.h>

OBJC_CLASS WKGroupSessionObserver;

namespace WebKit {

class WebPageProxy;

class GroupActivitiesSessionNotifier : public RefCountedAndCanMakeWeakPtr<GroupActivitiesSessionNotifier> {
    WTF_MAKE_TZONE_ALLOCATED(GroupActivitiesSessionNotifier);
public:
    static GroupActivitiesSessionNotifier& singleton();
    static Ref<GroupActivitiesSessionNotifier> create();

    bool hasSessionForURL(const URL&);
    RefPtr<GroupActivitiesSession> takeSessionForURL(const URL&);
    void removeSession(const GroupActivitiesSession&);

    void addWebPage(WebPageProxy&);
    void removeWebPage(WebPageProxy&);
    void webPageURLChanged(WebPageProxy&);

private:
    friend class NeverDestroyed<GroupActivitiesSessionNotifier>;
    GroupActivitiesSessionNotifier();

    void sessionStateChanged(const GroupActivitiesSession&, GroupActivitiesSession::State);

    HashMap<URL, Ref<GroupActivitiesSession>> m_sessions;
    RetainPtr<WKGroupSessionObserver> m_sessionObserver;
    WeakHashSet<WebPageProxy> m_webPages;
    GroupActivitiesSession::StateChangeObserver m_stateChangeObserver;
};

}

#endif
