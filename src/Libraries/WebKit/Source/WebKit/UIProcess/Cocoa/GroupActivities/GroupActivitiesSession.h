/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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

#include <wtf/Observer.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>
#include <wtf/WeakHashSet.h>

OBJC_CLASS WKGroupSession;

namespace WebKit {

class GroupActivitiesSession : public RefCounted<GroupActivitiesSession> {
    WTF_MAKE_TZONE_ALLOCATED(GroupActivitiesSession);
public:
    static Ref<GroupActivitiesSession> create(RetainPtr<WKGroupSession>&&);
    ~GroupActivitiesSession();

    void join();
    void leave();

    enum class State : uint8_t {
        Waiting,
        Joined,
        Invalidated,
    };
    State state() const;

    String uuid() const;
    URL fallbackURL() const;

    using StateChangeObserver = WTF::Observer<void(const GroupActivitiesSession&, State)>;
    void addStateChangeObserver(const StateChangeObserver&);

    using FallbackURLObserver = WTF::Observer<void(const GroupActivitiesSession&, URL)>;
    void addFallbackURLObserver(const FallbackURLObserver&);

private:
    friend class GroupActivitiesSessionNotifier;
    friend class GroupActivitiesCoordinator;
    GroupActivitiesSession(RetainPtr<WKGroupSession>&&);
    WKGroupSession* groupSession() { return m_groupSession.get(); }

    RetainPtr<WKGroupSession> m_groupSession;
    WeakHashSet<StateChangeObserver> m_stateChangeObservers;
    WeakHashSet<FallbackURLObserver> m_fallbackURLObservers;
};

}

#endif
