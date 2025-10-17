/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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
#include "RemoteMediaSourceProxy.h"

#if ENABLE(GPU_PROCESS) && ENABLE(MEDIA_SOURCE)

#include "GPUConnectionToWebProcess.h"
#include "MediaSourcePrivateRemoteMessageReceiverMessages.h"
#include "RemoteMediaPlayerManagerProxy.h"
#include "RemoteMediaPlayerProxy.h"
#include "RemoteMediaSourceProxyMessages.h"
#include "RemoteSourceBufferProxy.h"
#include <WebCore/ContentType.h>
#include <WebCore/NotImplemented.h>
#include <WebCore/SourceBufferPrivate.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteMediaSourceProxy);

RemoteMediaSourceProxy::RemoteMediaSourceProxy(RemoteMediaPlayerManagerProxy& manager, RemoteMediaSourceIdentifier identifier, RemoteMediaPlayerProxy& remoteMediaPlayerProxy)
    : m_manager(manager)
    , m_identifier(identifier)
    , m_remoteMediaPlayerProxy(remoteMediaPlayerProxy)
{
    ASSERT(RunLoop::isMain());

    connectionToWebProcess()->messageReceiverMap().addMessageReceiver(Messages::RemoteMediaSourceProxy::messageReceiverName(), m_identifier.toUInt64(), *this);
    manager.registerMediaSource(m_identifier, *this);
}

RemoteMediaSourceProxy::~RemoteMediaSourceProxy()
{
    disconnect();
}

void RemoteMediaSourceProxy::setMediaPlayers(RemoteMediaPlayerProxy& remoteMediaPlayerProxy, WebCore::MediaPlayerPrivateInterface* mediaPlayerPrivate)
{
    m_remoteMediaPlayerProxy = remoteMediaPlayerProxy;
    for (auto& sourceBuffer : m_sourceBuffers)
        sourceBuffer->setMediaPlayer(remoteMediaPlayerProxy);
    if (RefPtr protectedPrivate = m_private)
        protectedPrivate->setPlayer(mediaPlayerPrivate);
}

void RemoteMediaSourceProxy::disconnect()
{
    RefPtr connection = connectionToWebProcess();
    if (!connection)
        return;

    connection->messageReceiverMap().removeMessageReceiver(Messages::RemoteMediaSourceProxy::messageReceiverName(), m_identifier.toUInt64());
    m_manager = nullptr;
}

void RemoteMediaSourceProxy::setPrivateAndOpen(Ref<MediaSourcePrivate>&& mediaSourcePrivate)
{
    ASSERT(!m_private);
    m_private = WTFMove(mediaSourcePrivate);
}

void RemoteMediaSourceProxy::reOpen()
{
    ASSERT(m_private);
}

Ref<MediaTimePromise> RemoteMediaSourceProxy::waitForTarget(const SeekTarget& target)
{
    if (RefPtr connection = connectionToWebProcess())
        return connection->protectedConnection()->sendWithPromisedReply<MediaPromiseConverter>(Messages::MediaSourcePrivateRemoteMessageReceiver::ProxyWaitForTarget(target), m_identifier);

    return MediaTimePromise::createAndReject(PlatformMediaError::IPCError);
}

Ref<MediaPromise> RemoteMediaSourceProxy::seekToTime(const MediaTime& time)
{
    if (RefPtr connection = connectionToWebProcess())
        return connection->protectedConnection()->sendWithPromisedReply<MediaPromiseConverter>(Messages::MediaSourcePrivateRemoteMessageReceiver::ProxySeekToTime(time), m_identifier);

    return MediaPromise::createAndReject(PlatformMediaError::IPCError);
}

#if !RELEASE_LOG_DISABLED
void RemoteMediaSourceProxy::setLogIdentifier(uint64_t)
{
    notImplemented();
}
#endif

void RemoteMediaSourceProxy::failedToCreateRenderer(RendererType)
{
    notImplemented();
}

void RemoteMediaSourceProxy::addSourceBuffer(const WebCore::ContentType& contentType, AddSourceBufferCallback&& callback)
{
    RefPtr connection = connectionToWebProcess();
    if (!m_remoteMediaPlayerProxy || !connection)
        return;

    RefPtr<SourceBufferPrivate> sourceBufferPrivate;
    MediaSourcePrivate::AddStatus status = mediaSourcePrivate()->addSourceBuffer(contentType, sourceBufferPrivate);

    std::optional<RemoteSourceBufferIdentifier> remoteSourceIdentifier;
    if (status == MediaSourcePrivate::AddStatus::Ok) {
        auto identifier = RemoteSourceBufferIdentifier::generate();
        Ref remoteMediaPlayerProxy { *m_remoteMediaPlayerProxy };
        auto remoteSourceBufferProxy = RemoteSourceBufferProxy::create(*connection, identifier, sourceBufferPrivate.releaseNonNull(), remoteMediaPlayerProxy);
        m_sourceBuffers.append(WTFMove(remoteSourceBufferProxy));
        remoteSourceIdentifier = identifier;
    }

    callback(status, remoteSourceIdentifier);
}

void RemoteMediaSourceProxy::durationChanged(const MediaTime& duration)
{
    if (RefPtr protectedPrivate = m_private)
        protectedPrivate->durationChanged(duration);
}

void RemoteMediaSourceProxy::bufferedChanged(WebCore::PlatformTimeRanges&& buffered)
{
    if (RefPtr protectedPrivate = m_private)
        protectedPrivate->bufferedChanged(WTFMove(buffered));
}

void RemoteMediaSourceProxy::markEndOfStream(WebCore::MediaSourcePrivate::EndOfStreamStatus status )
{
    if (RefPtr protectedPrivate = m_private)
        protectedPrivate->markEndOfStream(status);
}

void RemoteMediaSourceProxy::unmarkEndOfStream()
{
    if (RefPtr protectedPrivate = m_private)
        protectedPrivate->unmarkEndOfStream();
}


void RemoteMediaSourceProxy::setMediaPlayerReadyState(WebCore::MediaPlayerEnums::ReadyState readyState)
{
    if (RefPtr protectedPrivate = m_private)
        protectedPrivate->setMediaPlayerReadyState(readyState);
}

void RemoteMediaSourceProxy::setTimeFudgeFactor(const MediaTime& fudgeFactor)
{
    if (RefPtr protectedPrivate = m_private)
        protectedPrivate->setTimeFudgeFactor(fudgeFactor);
}

void RemoteMediaSourceProxy::attached()
{
}

void RemoteMediaSourceProxy::shutdown()
{
    ASSERT(RunLoop::isMain());

    disconnect();

    if (RefPtr manager = m_manager.get())
        manager->invalidateMediaSource(m_identifier);
}

RefPtr<GPUConnectionToWebProcess> RemoteMediaSourceProxy::connectionToWebProcess() const
{
    ASSERT(RunLoop::isMain());

    RefPtr manager = m_manager.get();
    return manager ? manager->gpuConnectionToWebProcess() : nullptr;
}

std::optional<SharedPreferencesForWebProcess> RemoteMediaSourceProxy::sharedPreferencesForWebProcess() const
{
    if (RefPtr connection = connectionToWebProcess())
        return connection->sharedPreferencesForWebProcess();

    return std::nullopt;
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(MEDIA_SOURCE)
