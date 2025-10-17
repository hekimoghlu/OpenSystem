/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 7, 2022.
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

#include <WebCore/ExceptionData.h>
#include <WebCore/MediaSessionCoordinatorPrivate.h>
#include <wtf/RefPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
struct ExceptionData;
}

namespace WebKit {

using MediaSessionCommandCompletionHandler = CompletionHandler<void(std::optional<WebCore::ExceptionData>&&)>;

class MediaSessionCoordinatorProxyPrivate
    : public CanMakeWeakPtr<MediaSessionCoordinatorProxyPrivate>
    , public RefCounted<MediaSessionCoordinatorProxyPrivate> {
public:
    virtual ~MediaSessionCoordinatorProxyPrivate() = default;

    virtual String identifier() const = 0;

    virtual void join(MediaSessionCommandCompletionHandler&&) = 0;
    virtual void leave() = 0;
    virtual void seekTo(double, MediaSessionCommandCompletionHandler&&) = 0;
    virtual void play(MediaSessionCommandCompletionHandler&&) = 0;
    virtual void pause(MediaSessionCommandCompletionHandler&&) = 0;
    virtual void setTrack(const String&, MediaSessionCommandCompletionHandler&&) = 0;

    virtual void positionStateChanged(const std::optional<WebCore::MediaPositionState>&) = 0;
    virtual void readyStateChanged(WebCore::MediaSessionReadyState) = 0;
    virtual void playbackStateChanged(WebCore::MediaSessionPlaybackState) = 0;
    virtual void trackIdentifierChanged(const String&) = 0;

    virtual void setClient(WeakPtr<WebCore::MediaSessionCoordinatorClient> client) { m_client = client; }

protected:
    explicit MediaSessionCoordinatorProxyPrivate() = default;

    WeakPtr<WebCore::MediaSessionCoordinatorClient> client() const { return m_client; }

private:
    WeakPtr<WebCore::MediaSessionCoordinatorClient> m_client;
};

} // namespace WebKit

#endif // ENABLE(MEDIA_SESSION_COORDINATOR)
