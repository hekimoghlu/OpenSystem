/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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
#include "MediaSourceInterfaceMainThread.h"

#if ENABLE(MEDIA_SOURCE)

#include "ManagedMediaSource.h"
#include "MediaSource.h"
#include "SourceBufferList.h"
#include "TimeRanges.h"

namespace WebCore {

MediaSourceInterfaceMainThread::MediaSourceInterfaceMainThread(Ref<MediaSource>&& mediaSource)
    : m_mediaSource(WTFMove(mediaSource))
{
}

RefPtr<MediaSourcePrivateClient> MediaSourceInterfaceMainThread::client() const
{
    return m_mediaSource->client();
}

void MediaSourceInterfaceMainThread::monitorSourceBuffers()
{
    m_mediaSource->monitorSourceBuffers();
}

bool MediaSourceInterfaceMainThread::isClosed() const
{
    return m_mediaSource->isClosed();
}

MediaTime MediaSourceInterfaceMainThread::duration() const
{
    return m_mediaSource->duration();
}

PlatformTimeRanges MediaSourceInterfaceMainThread::buffered() const
{
    return m_mediaSource->buffered();
}

Ref<TimeRanges> MediaSourceInterfaceMainThread::seekable() const
{
    return m_mediaSource->seekable();
}

bool MediaSourceInterfaceMainThread::isStreamingContent() const
{
    if (RefPtr managedMediasource = dynamicDowncast<ManagedMediaSource>(m_mediaSource))
        return managedMediasource && managedMediasource->streamingAllowed() && managedMediasource->streaming();
    // We can assume that if we have active source buffers, later networking activity (such as stream or XHR requests) will be media related.
    return m_mediaSource->activeSourceBuffers()->length();
}

bool MediaSourceInterfaceMainThread::attachToElement(WeakPtr<HTMLMediaElement>&& element)
{
    return m_mediaSource->attachToElement(WTFMove(element));
}

void MediaSourceInterfaceMainThread::detachFromElement()
{
    m_mediaSource->detachFromElement();
}

void MediaSourceInterfaceMainThread::elementIsShuttingDown()
{
    m_mediaSource->elementIsShuttingDown();
}

void MediaSourceInterfaceMainThread::openIfDeferredOpen()
{
    m_mediaSource->openIfDeferredOpen();
}

bool MediaSourceInterfaceMainThread::isManaged() const
{
    return m_mediaSource->isManaged();
}

void MediaSourceInterfaceMainThread::setAsSrcObject(bool set)
{
    m_mediaSource->setAsSrcObject(set);
}

void MediaSourceInterfaceMainThread::memoryPressure()
{
    m_mediaSource->memoryPressure();
}

bool MediaSourceInterfaceMainThread::detachable() const
{
    return m_mediaSource->detachable();
}

} // namespace WebCore

#endif // ENABLE(MEDIA_SOURCE)
