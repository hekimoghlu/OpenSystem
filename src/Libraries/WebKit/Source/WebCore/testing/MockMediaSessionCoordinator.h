/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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

#if ENABLE(MEDIA_SESSION_COORDINATOR)

#include "MediaSessionCoordinatorPrivate.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class ScriptExecutionContext;
class StringCallback;

class MockMediaSessionCoordinator : public MediaSessionCoordinatorPrivate, public CanMakeWeakPtr<MockMediaSessionCoordinator> {
    WTF_MAKE_TZONE_ALLOCATED(MockMediaSessionCoordinator);
public:
    static Ref<MockMediaSessionCoordinator> create(ScriptExecutionContext&, RefPtr<StringCallback>&&);

    void setCommandsShouldFail(bool shouldFail) { m_failCommands = shouldFail; }

private:
    MockMediaSessionCoordinator(ScriptExecutionContext&, RefPtr<StringCallback>&&);

    String identifier() const final { return "Mock Coordinator"_s; }

    void join(CompletionHandler<void(std::optional<Exception>&&)>&&) final;
    void leave() final;

    void seekTo(double, CompletionHandler<void(std::optional<Exception>&&)>&&) final;
    void play(CompletionHandler<void(std::optional<Exception>&&)>&&) final;
    void pause(CompletionHandler<void(std::optional<Exception>&&)>&&) final;
    void setTrack(const String&, CompletionHandler<void(std::optional<Exception>&&)>&&) final;

    void positionStateChanged(const std::optional<MediaPositionState>&) final;
    void readyStateChanged(MediaSessionReadyState) final;
    void playbackStateChanged(MediaSessionPlaybackState) final;
    void trackIdentifierChanged(const String&) final;

    ASCIILiteral logClassName() const { return "MockMediaSessionCoordinator"_s; }
    WTFLogChannel& logChannel() const;

    std::optional<Exception> result() const;

    Ref<ScriptExecutionContext> m_context;
    RefPtr<StringCallback> m_stateChangeListener;
    bool m_failCommands { false };
};

}

#endif // ENABLE(MEDIA_SESSION_COORDINATOR)
