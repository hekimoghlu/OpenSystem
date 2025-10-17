/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 17, 2023.
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
#import "GroupActivitiesSession.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(MEDIA_SESSION_COORDINATOR) && HAVE(GROUP_ACTIVITIES)

#import "WKGroupSession.h"

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(GroupActivitiesSession);

Ref<GroupActivitiesSession> GroupActivitiesSession::create(RetainPtr<WKGroupSession>&& session)
{
    return adoptRef(*new GroupActivitiesSession(WTFMove(session)));
}

GroupActivitiesSession::GroupActivitiesSession(RetainPtr<WKGroupSession>&& session)
    : m_groupSession(WTFMove(session))
{
    m_groupSession.get().newActivityCallback = [this] (WKURLActivity *activity) {
        m_fallbackURLObservers.forEach([this, activity] (auto& observer) {
            observer(*this, activity.fallbackURL);
        });
    };

    m_groupSession.get().stateChangedCallback = [this] (WKGroupSessionState state) {
        static_assert(static_cast<size_t>(State::Waiting) == static_cast<size_t>(WKGroupSessionStateWaiting), "MediaSessionCoordinatorState::Waiting != WKGroupSessionStateWaiting");
        static_assert(static_cast<size_t>(State::Joined) == static_cast<size_t>(WKGroupSessionStateJoined), "MediaSessionCoordinatorState::Joined != WKGroupSessionStateJoined");
        static_assert(static_cast<size_t>(State::Invalidated) == static_cast<size_t>(WKGroupSessionStateInvalidated), "MediaSessionCoordinatorState::Closed != WKGroupSessionStateInvalidated");
        m_stateChangeObservers.forEach([this, state] (auto& observer) {
            observer(*this, static_cast<State>(state));
        });
    };
}

GroupActivitiesSession::~GroupActivitiesSession()
{
    m_groupSession.get().newActivityCallback = nil;
    m_groupSession.get().stateChangedCallback = nil;
}

void GroupActivitiesSession::join()
{
    [m_groupSession join];
}

void GroupActivitiesSession::leave()
{
    [m_groupSession leave];
}

auto GroupActivitiesSession::state() const -> State
{
    static_assert(static_cast<size_t>(State::Waiting) == static_cast<size_t>(WKGroupSessionStateWaiting), "State::Waiting != WKGroupSessionStateWaiting");
    static_assert(static_cast<size_t>(State::Joined) == static_cast<size_t>(WKGroupSessionStateJoined), "State::Joined != WKGroupSessionStateJoined");
    static_assert(static_cast<size_t>(State::Invalidated) == static_cast<size_t>(WKGroupSessionStateInvalidated), "State::Invalidated != WKGroupSessionStateInvalidated");
    return static_cast<State>([m_groupSession state]);
}

String GroupActivitiesSession::uuid() const
{
    return [m_groupSession uuid].UUIDString;
}

URL GroupActivitiesSession::fallbackURL() const
{
    return [m_groupSession activity].fallbackURL;
}

void GroupActivitiesSession::addStateChangeObserver(const StateChangeObserver& observer)
{
    m_stateChangeObservers.add(observer);
}

void GroupActivitiesSession::addFallbackURLObserver(const FallbackURLObserver& observer)
{
    m_fallbackURLObservers.add(observer);
}

}

#endif
