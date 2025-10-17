/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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
#include "MediaSessionCoordinatorProxyPrivate.h"
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS AVDelegatingPlaybackCoordinator;
OBJC_CLASS AVDelegatingPlaybackCoordinatorPlayCommand;
OBJC_CLASS AVDelegatingPlaybackCoordinatorPauseCommand;
OBJC_CLASS AVDelegatingPlaybackCoordinatorSeekCommand;
OBJC_CLASS AVDelegatingPlaybackCoordinatorBufferingCommand;
OBJC_CLASS AVDelegatingPlaybackCoordinatorPrepareTransitionCommand;
OBJC_CLASS WKGroupActivitiesCoordinatorDelegate;

namespace WebKit {

class GroupActivitiesCoordinator final : public MediaSessionCoordinatorProxyPrivate {
    WTF_MAKE_TZONE_ALLOCATED(GroupActivitiesCoordinator);
public:
    static Ref<GroupActivitiesCoordinator> create(GroupActivitiesSession&);
    ~GroupActivitiesCoordinator();

    using CommandCompletionHandler = Function<void()>;
    void issuePlayCommand(AVDelegatingPlaybackCoordinatorPlayCommand *, CommandCompletionHandler&&);
    void issuePauseCommand(AVDelegatingPlaybackCoordinatorPauseCommand *, CommandCompletionHandler&&);
    void issueSeekCommand(AVDelegatingPlaybackCoordinatorSeekCommand *, CommandCompletionHandler&&);
    void issueBufferingCommand(AVDelegatingPlaybackCoordinatorBufferingCommand *, CommandCompletionHandler&&);
    void issuePrepareTransitionCommand(AVDelegatingPlaybackCoordinatorPrepareTransitionCommand *);

private:
    GroupActivitiesCoordinator(GroupActivitiesSession&);

    void sessionStateChanged(const GroupActivitiesSession&, GroupActivitiesSession::State);

    using CoordinatorCompletionHandler = CompletionHandler<void(std::optional<WebCore::ExceptionData>&&)>;
    String identifier() const final;
    void join(CoordinatorCompletionHandler&&) final;
    void leave() final;
    void seekTo(double, CoordinatorCompletionHandler&&) final;
    void play(CoordinatorCompletionHandler&&) final;
    void pause(CoordinatorCompletionHandler&&) final;
    void setTrack(const String&, CoordinatorCompletionHandler&&) final;
    void positionStateChanged(const std::optional<WebCore::MediaPositionState>&) final;
    void readyStateChanged(WebCore::MediaSessionReadyState) final;
    void playbackStateChanged(WebCore::MediaSessionPlaybackState) final;
    void trackIdentifierChanged(const String&) final;

    Ref<GroupActivitiesSession> m_session;
    RetainPtr<WKGroupActivitiesCoordinatorDelegate> m_delegate;
    RetainPtr<AVDelegatingPlaybackCoordinator> m_playbackCoordinator;

    std::optional<WebCore::MediaPositionState> m_positionState;
    std::optional<WebCore::MediaSessionReadyState> m_readyState;
    std::optional<WebCore::MediaSessionPlaybackState> m_playbackState;

    GroupActivitiesSession::StateChangeObserver m_stateChangeObserver;
};

}

#endif
