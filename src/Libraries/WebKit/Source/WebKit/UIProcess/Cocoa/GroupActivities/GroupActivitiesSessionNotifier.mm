/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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
#import "config.h"
#import "GroupActivitiesSessionNotifier.h"

#if ENABLE(MEDIA_SESSION_COORDINATOR) && HAVE(GROUP_ACTIVITIES)

#import "GroupActivitiesCoordinator.h"
#import "WKGroupSession.h"
#import "WebFrameProxy.h"
#import "WebPageProxy.h"
#import <mutex>
#import <wtf/TZoneMallocInlines.h>

#import "WebKitSwiftSoftLink.h"

namespace WebKit {

using namespace PAL;
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(GroupActivitiesSessionNotifier);

GroupActivitiesSessionNotifier& GroupActivitiesSessionNotifier::singleton()
{
    static NeverDestroyed<GroupActivitiesSessionNotifier> notifier;
    return notifier;
}

GroupActivitiesSessionNotifier::GroupActivitiesSessionNotifier()
    : m_sessionObserver(adoptNS([allocWKGroupSessionObserverInstance() init]))
    , m_stateChangeObserver([this] (auto& session, auto state) { sessionStateChanged(session, state); })
{
    m_sessionObserver.get().newSessionCallback = [this, weakThis = WeakPtr { *this }] (WKGroupSession *groupSession) {
        RefPtr protectedThis = weakThis.get();
        if (!protectedThis)
            return;

        auto session = GroupActivitiesSession::create(groupSession);
        session->addStateChangeObserver(m_stateChangeObserver);

        for (auto& page : copyToVector(m_webPages)) {
            if (page->mainFrame() && page->mainFrame()->url() == session->fallbackURL()) {
                auto coordinator = GroupActivitiesCoordinator::create(session);
                page->createMediaSessionCoordinator(WTFMove(coordinator), [] (bool) { });
                return;
            }
        }

        auto result = m_sessions.add(session->fallbackURL(), session.copyRef());
        ASSERT_UNUSED(result, result.isNewEntry);

        [[NSWorkspace sharedWorkspace] openURL:session->fallbackURL()];
    };
}

void GroupActivitiesSessionNotifier::sessionStateChanged(const GroupActivitiesSession& session, GroupActivitiesSession::State state)
{
    if (state == GroupActivitiesSession::State::Invalidated)
        m_sessions.remove(session.fallbackURL());
};

void GroupActivitiesSessionNotifier::addWebPage(WebPageProxy& webPage)
{
    ASSERT(!m_webPages.contains(webPage));
    m_webPages.add(webPage);

    RefPtr frame = webPage.mainFrame();
    if (!frame)
        return;

    auto session = takeSessionForURL(frame->url());
    if (!session)
        return;

    auto coordinator = GroupActivitiesCoordinator::create(*session);
    webPage.createMediaSessionCoordinator(WTFMove(coordinator), [] (bool) { });
}

void GroupActivitiesSessionNotifier::removeWebPage(WebPageProxy& webPage)
{
    ASSERT(m_webPages.contains(webPage));
    m_webPages.remove(webPage);
}

void GroupActivitiesSessionNotifier::webPageURLChanged(WebPageProxy& webPage)
{
    ASSERT(m_webPages.contains(webPage));
    if (!m_webPages.contains(webPage))
        return;

    auto frame = webPage.mainFrame();
    if (!frame)
        return;

    auto session = takeSessionForURL(frame->url());
    if (!session)
        return;

    auto coordinator = GroupActivitiesCoordinator::create(*session);
    webPage.createMediaSessionCoordinator(WTFMove(coordinator), [] (bool) { });
}

bool GroupActivitiesSessionNotifier::hasSessionForURL(const URL& url)
{
    return m_sessions.contains(url);
}

RefPtr<GroupActivitiesSession> GroupActivitiesSessionNotifier::takeSessionForURL(const URL& url)
{
    return m_sessions.take(url);
}

}

#endif
