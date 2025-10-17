/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 14, 2023.
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

#include "ActiveDOMObject.h"
#include "EventLoop.h"
#include "UserGestureIndicator.h"
#include <memory>
#include <wtf/MonotonicTime.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/Seconds.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class DOMTimerFireState;
class Document;
class ImminentlyScheduledWorkScope;
class ScheduledAction;

class DOMTimer final : public RefCountedAndCanMakeWeakPtr<DOMTimer>, public ActiveDOMObject {
    WTF_MAKE_NONCOPYABLE(DOMTimer);
    WTF_MAKE_TZONE_ALLOCATED(DOMTimer);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    WEBCORE_EXPORT virtual ~DOMTimer();

    static Seconds defaultMinimumInterval() { return 4_ms; }
    static Seconds defaultAlignmentInterval() { return 0_s; }
    static Seconds defaultAlignmentIntervalInLowPowerOrThermallyMitigatedMode() { return 30_ms; }
    static Seconds nonInteractedCrossOriginFrameAlignmentInterval() { return 30_ms; }
    static Seconds hiddenPageAlignmentInterval() { return 1_s; }

    enum class Type : bool { SingleShot, Repeating };
    static int install(ScriptExecutionContext&, std::unique_ptr<ScheduledAction>, Seconds timeout, Type);
    static int install(ScriptExecutionContext&, Function<void(ScriptExecutionContext&)>&&, Seconds timeout, Type);
    static void removeById(ScriptExecutionContext&, int timeoutId);

    // Notify that the interval may need updating (e.g. because the minimum interval
    // setting for the context has changed).
    void updateTimerIntervalIfNecessary();

    static void scriptDidInteractWithPlugin();

    EventLoopTimerHandle timer() const { return m_timer; }
    bool hasReachedMaxNestingLevel() const { return m_hasReachedMaxNestingLevel; }

private:
    DOMTimer(ScriptExecutionContext&, Function<void(ScriptExecutionContext&)>&&, Seconds interval, Type);
    friend class Internals;

    WEBCORE_EXPORT Seconds intervalClampedToMinimum() const;

    bool isDOMTimersThrottlingEnabled(const Document&) const;
    void updateThrottlingStateIfNecessary(const DOMTimerFireState&);

    void fired();

    // ActiveDOMObject.
    void stop() final;

    void makeImminentlyScheduledWorkScopeIfPossible(ScriptExecutionContext&);
    void clearImminentlyScheduledWorkScope();

    enum TimerThrottleState {
        Undetermined,
        ShouldThrottle,
        ShouldNotThrottle
    };

    int m_timeoutId;
    int m_nestingLevel;
    EventLoopTimerHandle m_timer;
    Function<void(ScriptExecutionContext&)> m_action;
    Seconds m_originalInterval;
    TimerThrottleState m_throttleState;
    bool m_oneShot;
    bool m_hasReachedMaxNestingLevel;
    Seconds m_currentTimerInterval;
    RefPtr<UserGestureToken> m_userGestureTokenToForward;
    RefPtr<ImminentlyScheduledWorkScope> m_imminentlyScheduledWorkScope;
};

} // namespace WebCore
