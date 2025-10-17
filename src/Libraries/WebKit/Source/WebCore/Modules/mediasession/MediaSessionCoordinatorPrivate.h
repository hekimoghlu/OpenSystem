/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 31, 2025.
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

#include "Exception.h"
#include "MediaPositionState.h"
#include "MediaSessionCoordinatorState.h"
#include "MediaSessionPlaybackState.h"
#include "MediaSessionReadyState.h"
#include <wtf/MonotonicTime.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class MediaSessionCoordinatorClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::MediaSessionCoordinatorClient> : std::true_type { };
}

namespace WTF {
class Logger;
}

namespace WebCore {

class MediaSessionCoordinatorClient : public CanMakeWeakPtr<MediaSessionCoordinatorClient> {
public:
    virtual ~MediaSessionCoordinatorClient() = default;

    virtual void seekSessionToTime(double, CompletionHandler<void(bool)>&&) = 0;
    virtual void playSession(std::optional<double> atTime, std::optional<MonotonicTime> hostTime, CompletionHandler<void(bool)>&&) = 0;
    virtual void pauseSession(CompletionHandler<void(bool)>&&) = 0;
    virtual void setSessionTrack(const String&, CompletionHandler<void(bool)>&&) = 0;
    virtual void coordinatorStateChanged(WebCore::MediaSessionCoordinatorState) = 0;
};

class MediaSessionCoordinatorPrivate : public RefCounted<MediaSessionCoordinatorPrivate> {
public:
    virtual ~MediaSessionCoordinatorPrivate() = default;

    virtual String identifier() const = 0;

    virtual void join(CompletionHandler<void(std::optional<Exception>&&)>&&) = 0;
    virtual void leave() = 0;

    virtual void seekTo(double, CompletionHandler<void(std::optional<Exception>&&)>&&) = 0;
    virtual void play(CompletionHandler<void(std::optional<Exception>&&)>&&) = 0;
    virtual void pause(CompletionHandler<void(std::optional<Exception>&&)>&&) = 0;
    virtual void setTrack(const String&, CompletionHandler<void(std::optional<Exception>&&)>&&) = 0;

    virtual void positionStateChanged(const std::optional<MediaPositionState>&) = 0;
    virtual void readyStateChanged(MediaSessionReadyState) = 0;
    virtual void playbackStateChanged(MediaSessionPlaybackState) = 0;
    virtual void trackIdentifierChanged(const String&) = 0;

    void setLogger(const Logger&, uint64_t);
    virtual void setClient(WeakPtr<MediaSessionCoordinatorClient> client) { m_client = client;}

protected:
    explicit MediaSessionCoordinatorPrivate() = default;

    const Logger* loggerPtr() const { return m_logger.get(); }
    uint64_t logIdentifier() const { return m_logIdentifier; }

    WeakPtr<MediaSessionCoordinatorClient> client() const { return m_client; }

private:
    RefPtr<const Logger> m_logger;
    uint64_t m_logIdentifier { 0 };
    WeakPtr<MediaSessionCoordinatorClient> m_client;
};

}

#endif // ENABLE(MEDIA_SESSION_COORDINATOR)
