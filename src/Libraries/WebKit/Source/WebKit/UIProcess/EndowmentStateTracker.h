/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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

#if PLATFORM(IOS_FAMILY)

#import <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakHashSet.h>

OBJC_CLASS NSSet;
OBJC_CLASS RBSProcessMonitor;

namespace WebKit {

class WebPageProxy;

class EndowmentStateTrackerClient : public AbstractRefCountedAndCanMakeWeakPtr<EndowmentStateTrackerClient> {
public:
    virtual ~EndowmentStateTrackerClient() = default;
    virtual void isUserFacingChanged(bool) { }
    virtual void isVisibleChanged(bool) { }
};

class EndowmentStateTracker {
    WTF_MAKE_TZONE_ALLOCATED(EndowmentStateTracker);
public:
    static EndowmentStateTracker& singleton();

    bool isVisible() const { return ensureState().isVisible; }
    bool isUserFacing() const { return ensureState().isUserFacing; }

    void addClient(EndowmentStateTrackerClient&);
    void removeClient(EndowmentStateTrackerClient&);

    static bool isApplicationForeground(pid_t);

private:
    friend class NeverDestroyed<EndowmentStateTracker>;
    EndowmentStateTracker() = default;

    void registerMonitorIfNecessary();

    struct State {
        bool isUserFacing;
        bool isVisible;
    };
    static State stateFromEndowments(NSSet *endowments);
    const State& ensureState() const;
    void setState(State&&);

    WeakHashSet<EndowmentStateTrackerClient> m_clients;
    RetainPtr<RBSProcessMonitor> m_processMonitor;
    mutable std::optional<State> m_state;
};

}

#endif
