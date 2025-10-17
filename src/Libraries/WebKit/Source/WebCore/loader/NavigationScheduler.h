/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 1, 2023.
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

#include "FrameLoaderTypes.h"
#include "LoaderMalloc.h"
#include "Timer.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class Document;
class FormSubmission;
class Frame;
class LocalFrame;
class ScheduledNavigation;
class SecurityOrigin;

enum class NewLoadInProgress : bool { No, Yes };
enum class ScheduleLocationChangeResult : uint8_t { Stopped, Completed, Started };
enum class ScheduleHistoryNavigationResult : bool { Completed, Aborted };

class NavigationScheduler final {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    explicit NavigationScheduler(Frame&);
    ~NavigationScheduler();

    void ref() const;
    void deref() const;

    bool redirectScheduledDuringLoad();
    bool locationChangePending();

    void scheduleRedirect(Document& initiatingDocument, double delay, const URL&, IsMetaRefresh);
    void scheduleLocationChange(Document& initiatingDocument, SecurityOrigin&, const URL&, const String& referrer, LockHistory = LockHistory::Yes, LockBackForwardList = LockBackForwardList::Yes, NavigationHistoryBehavior historyHandling = NavigationHistoryBehavior::Auto, CompletionHandler<void(ScheduleLocationChangeResult)>&& = [] (ScheduleLocationChangeResult) { });
    void scheduleFormSubmission(Ref<FormSubmission>&&);
    void scheduleRefresh(Document& initiatingDocument);
    void scheduleHistoryNavigation(int steps);
    void scheduleHistoryNavigationByKey(const String&key, CompletionHandler<void(ScheduleHistoryNavigationResult)>&&);
    void schedulePageBlock(Document& originDocument);

    void startTimer();

    void cancel(NewLoadInProgress = NewLoadInProgress::No);
    void clear();

    bool hasQueuedNavigation() const;

private:
    bool shouldScheduleNavigation() const;
    bool shouldScheduleNavigation(const URL&) const;

    void timerFired();
    void schedule(std::unique_ptr<ScheduledNavigation>);
    Ref<Frame> protectedFrame() const;

    static LockBackForwardList mustLockBackForwardList(Frame& targetFrame);

    WeakRef<Frame> m_frame;
    Timer m_timer;
    std::unique_ptr<ScheduledNavigation> m_redirect;
};

} // namespace WebCore
