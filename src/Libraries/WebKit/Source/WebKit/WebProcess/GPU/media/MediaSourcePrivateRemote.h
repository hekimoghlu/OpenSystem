/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 12, 2025.
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

#if ENABLE(GPU_PROCESS) && ENABLE(MEDIA_SOURCE)

#include "GPUProcessConnection.h"
#include "RemoteMediaPlayerMIMETypeCache.h"
#include "RemoteMediaSourceIdentifier.h"
#include "WorkQueueMessageReceiver.h"
#include <WebCore/ContentType.h>
#include <WebCore/MediaSourcePrivate.h>
#include <WebCore/MediaSourcePrivateClient.h>
#include <WebCore/SourceBufferPrivate.h>
#include <atomic>
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/LoggerHelper.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace IPC {
class Connection;
class Decoder;
}

namespace WebKit {

class MediaPlayerPrivateRemote;
class SourceBufferPrivateRemote;

class MediaSourcePrivateRemote final
    : public WebCore::MediaSourcePrivate
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
{
public:
    static Ref<MediaSourcePrivateRemote> create(GPUProcessConnection&, RemoteMediaSourceIdentifier, RemoteMediaPlayerMIMETypeCache&, const MediaPlayerPrivateRemote&, WebCore::MediaSourcePrivateClient&);
    virtual ~MediaSourcePrivateRemote();

    // MediaSourcePrivate overrides
    RefPtr<WebCore::MediaPlayerPrivateInterface> player() const final;
    constexpr WebCore::MediaPlatformType platformType() const final { return WebCore::MediaPlatformType::Remote; }
    AddStatus addSourceBuffer(const WebCore::ContentType&, RefPtr<WebCore::SourceBufferPrivate>&) final;
    void removeSourceBuffer(WebCore::SourceBufferPrivate&) final { }
    void notifyActiveSourceBuffersChanged() final { };
    void durationChanged(const MediaTime&) final;
    void markEndOfStream(EndOfStreamStatus) final;
    void unmarkEndOfStream() final;
    WebCore::MediaPlayer::ReadyState mediaPlayerReadyState() const final;
    void setMediaPlayerReadyState(WebCore::MediaPlayer::ReadyState) final;
    void setPlayer(WebCore::MediaPlayerPrivateInterface*) final;
    void shutdown() final;

    void setTimeFudgeFactor(const MediaTime&) final;

    RemoteMediaSourceIdentifier identifier() const { return m_identifier; }

    static WorkQueue& queue();

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger.get(); }
    uint64_t nextSourceBufferLogIdentifier() { return childLogIdentifier(m_logIdentifier, ++m_nextSourceBufferID); }
#endif

    class MessageReceiver : public IPC::WorkQueueMessageReceiver {
    public:
        static Ref<MessageReceiver> create(MediaSourcePrivateRemote& parent)
        {
            return adoptRef(*new MessageReceiver(parent));
        }

    private:
        MessageReceiver(MediaSourcePrivateRemote&);
        void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
        void proxyWaitForTarget(const WebCore::SeekTarget&, CompletionHandler<void(WebCore::MediaTimePromise::Result&&)>&&);
        void proxySeekToTime(const MediaTime&, CompletionHandler<void(WebCore::MediaPromise::Result&&)>&&);

        RefPtr<WebCore::MediaSourcePrivateClient> client() const;
        ThreadSafeWeakPtr<MediaSourcePrivateRemote> m_parent;
    };
private:
    friend class MessageReceiver;
    MediaSourcePrivateRemote(GPUProcessConnection&, RemoteMediaSourceIdentifier, RemoteMediaPlayerMIMETypeCache&, const MediaPlayerPrivateRemote&, WebCore::MediaSourcePrivateClient&);

    void bufferedChanged(const WebCore::PlatformTimeRanges&) final;

    void ensureOnDispatcherSync(Function<void()>&&) const;

    bool isGPURunning() const { return !m_shutdown; }

    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
    Ref<MessageReceiver> m_receiver;
    RemoteMediaSourceIdentifier m_identifier;
    CheckedRef<RemoteMediaPlayerMIMETypeCache> m_mimeTypeCache;
    ThreadSafeWeakPtr<MediaPlayerPrivateRemote> m_mediaPlayerPrivate;
    std::atomic<bool> m_shutdown { false };
    std::atomic<WebCore::MediaPlayer::ReadyState> m_mediaPlayerReadyState { WebCore::MediaPlayer::ReadyState::HaveNothing };

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const override { return "MediaSourcePrivateRemote"_s; }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    WTFLogChannel& logChannel() const final;

    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
    uint64_t m_nextSourceBufferID { 0 };
#endif
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::MediaSourcePrivateRemote)
static bool isType(const WebCore::MediaSourcePrivate& mediaSource) { return mediaSource.platformType() == WebCore::MediaPlatformType::Remote; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(GPU_PROCESS) && ENABLE(MEDIA_SOURCE)
