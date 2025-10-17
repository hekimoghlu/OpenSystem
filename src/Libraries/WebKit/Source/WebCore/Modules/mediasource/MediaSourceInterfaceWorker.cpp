/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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
#include "MediaSourceInterfaceWorker.h"

#if ENABLE(MEDIA_SOURCE_IN_WORKERS)

#include "ManagedMediaSource.h"
#include "MediaSource.h"
#include "MediaSourceHandle.h"
#include "MediaSourcePrivate.h"
#include "MediaSourcePrivateClient.h"
#include "PlatformTimeRanges.h"
#include "TimeRanges.h"

namespace WebCore {

MediaSourceInterfaceWorker::MediaSourceInterfaceWorker(Ref<MediaSourceHandle>&& handle)
    : m_handle(WTFMove(handle))
    , m_client(m_handle->mediaSourcePrivateClient())
{
}

RefPtr<MediaSourcePrivateClient> MediaSourceInterfaceWorker::client() const
{
    return m_client.get();
}

void MediaSourceInterfaceWorker::monitorSourceBuffers()
{
    ASSERT(m_handle->hasEverBeenAssignedAsSrcObject());
    m_handle->ensureOnDispatcher([](MediaSource& mediaSource) {
        mediaSource.monitorSourceBuffers();
    });
}

bool MediaSourceInterfaceWorker::isClosed() const
{
    if (RefPtr mediaSourcePrivate = m_handle->mediaSourcePrivate())
        return mediaSourcePrivate->readyState() == MediaSource::ReadyState::Closed;
    return true;
}

MediaTime MediaSourceInterfaceWorker::duration() const
{
    if (RefPtr mediaSourcePrivate = m_handle->mediaSourcePrivate(); mediaSourcePrivate && !isClosed())
        return mediaSourcePrivate->duration();
    return MediaTime::invalidTime();
}

PlatformTimeRanges MediaSourceInterfaceWorker::buffered() const
{
    if (RefPtr mediaSourcePrivate = m_handle->mediaSourcePrivate(); mediaSourcePrivate && !isClosed())
        return mediaSourcePrivate->buffered();
    return PlatformTimeRanges::emptyRanges();
}

Ref<TimeRanges> MediaSourceInterfaceWorker::seekable() const
{
    if (RefPtr mediaSourcePrivate = m_handle->mediaSourcePrivate(); mediaSourcePrivate && !isClosed())
        return TimeRanges::create(mediaSourcePrivate->seekable());
    return TimeRanges::create();
}

bool MediaSourceInterfaceWorker::isStreamingContent() const
{
    RefPtr mediaSourcePrivate = m_handle->mediaSourcePrivate();
    return mediaSourcePrivate && m_handle->isManaged() && mediaSourcePrivate->streamingAllowed() && mediaSourcePrivate->streaming();
}

bool MediaSourceInterfaceWorker::attachToElement(WeakPtr<HTMLMediaElement>&& element)
{
    if (m_handle->hasEverBeenAssignedAsSrcObject())
        return false;
    m_handle->setHasEverBeenAssignedAsSrcObject();
    bool forceRun = true;
    m_handle->ensureOnDispatcher([element = WTFMove(element)](MediaSource& mediaSource) mutable {
        mediaSource.attachToElement(WTFMove(element));
    }, forceRun);
    return true;
}

void MediaSourceInterfaceWorker::detachFromElement()
{
    ASSERT(m_handle->hasEverBeenAssignedAsSrcObject());
    m_handle->ensureOnDispatcher([](MediaSource& mediaSource) {
        mediaSource.detachFromElement();
    });
}

void MediaSourceInterfaceWorker::elementIsShuttingDown()
{
    ASSERT(m_handle->hasEverBeenAssignedAsSrcObject());
    m_handle->ensureOnDispatcher([](MediaSource& mediaSource) {
        mediaSource.elementIsShuttingDown();
    });
}

void MediaSourceInterfaceWorker::openIfDeferredOpen()
{
    ASSERT(m_handle->hasEverBeenAssignedAsSrcObject());
    m_handle->ensureOnDispatcher([](MediaSource& mediaSource) {
        mediaSource.openIfDeferredOpen();
    });
}

bool MediaSourceInterfaceWorker::isManaged() const
{
    return m_handle->isManaged();
}

void MediaSourceInterfaceWorker::setAsSrcObject(bool set)
{
    m_handle->ensureOnDispatcher([set](MediaSource& mediaSource) {
        mediaSource.setAsSrcObject(set);
    });
}

void MediaSourceInterfaceWorker::memoryPressure()
{
    m_handle->ensureOnDispatcher([](MediaSource& mediaSource) {
        mediaSource.memoryPressure();
    });
}

bool MediaSourceInterfaceWorker::detachable() const
{
    return m_handle->detachable();
}

} // namespace WebCore

#endif // ENABLE(MEDIA_SOURCE_IN_WORKERS)
