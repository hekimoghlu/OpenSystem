/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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
#include "config.h"
#include "MockMediaSessionCoordinator.h"

#if ENABLE(MEDIA_SESSION_COORDINATOR)

#include "Logging.h"
#include "ScriptExecutionContext.h"
#include "StringCallback.h"
#include <wtf/CompletionHandler.h>
#include <wtf/LoggerHelper.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/UUID.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/StringView.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MockMediaSessionCoordinator);

Ref<MockMediaSessionCoordinator> MockMediaSessionCoordinator::create(ScriptExecutionContext& context, RefPtr<StringCallback>&& listener)
{
    return adoptRef(*new MockMediaSessionCoordinator(context, WTFMove(listener)));
}

MockMediaSessionCoordinator::MockMediaSessionCoordinator(ScriptExecutionContext& context, RefPtr<StringCallback>&& listener)
    : m_context(context)
    , m_stateChangeListener(WTFMove(listener))
{
}

std::optional<Exception> MockMediaSessionCoordinator::result() const
{
    if (m_failCommands)
        return Exception { ExceptionCode::InvalidStateError };

    return std::nullopt;
}

void MockMediaSessionCoordinator::join(CompletionHandler<void(std::optional<Exception>&&)>&& callback)
{
    m_context->postTask([this, callback = WTFMove(callback)] (ScriptExecutionContext&) mutable {
        callback(result());
    });
}

void MockMediaSessionCoordinator::leave()
{
}

void MockMediaSessionCoordinator::seekTo(double time, CompletionHandler<void(std::optional<Exception>&&)>&& callback)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER, time);
    m_context->postTask([this, callback = WTFMove(callback)] (ScriptExecutionContext&) mutable {
        callback(result());
    });
}

void MockMediaSessionCoordinator::play(CompletionHandler<void(std::optional<Exception>&&)>&& callback)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER);
    m_context->postTask([this, callback = WTFMove(callback)] (ScriptExecutionContext&) mutable {
        callback(result());
    });
}

void MockMediaSessionCoordinator::pause(CompletionHandler<void(std::optional<Exception>&&)>&& callback)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER);
    m_context->postTask([this, callback = WTFMove(callback)] (ScriptExecutionContext&) mutable {
        callback(result());
    });
}

void MockMediaSessionCoordinator::setTrack(const String&, CompletionHandler<void(std::optional<Exception>&&)>&& callback)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER);
    m_context->postTask([this, callback = WTFMove(callback)] (ScriptExecutionContext&) mutable {
        callback(result());
    });
}

void MockMediaSessionCoordinator::positionStateChanged(const std::optional<MediaPositionState>&)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER);
    m_stateChangeListener->scheduleCallback(m_context.get(), "positionStateChanged"_s);
}

void MockMediaSessionCoordinator::readyStateChanged(MediaSessionReadyState state)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER, state);
    m_stateChangeListener->scheduleCallback(m_context.get(), "readyStateChanged"_s);
}

void MockMediaSessionCoordinator::playbackStateChanged(MediaSessionPlaybackState state)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER, state);
    m_stateChangeListener->scheduleCallback(m_context.get(), "playbackStateChanged"_s);
}

void MockMediaSessionCoordinator::trackIdentifierChanged(const String& identifier)
{
    ALWAYS_LOG_IF_POSSIBLE(LOGIDENTIFIER, identifier);
    m_stateChangeListener->scheduleCallback(m_context.get(), "trackIdentifierChanged"_s);
}

WTFLogChannel& MockMediaSessionCoordinator::logChannel() const
{
    return LogMedia;
}

}

#endif // ENABLE(MEDIA_SESSION_COORDINATOR)
