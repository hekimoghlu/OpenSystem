/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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

#if ENABLE(MEDIA_SOURCE) && USE(AVFOUNDATION)

#include "ProcessIdentity.h"
#include "MediaSourcePrivate.h"
#include "MediaSourcePrivateClient.h"
#include <wtf/Deque.h>
#include <wtf/LoggerHelper.h>
#include <wtf/RefPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/Vector.h>

OBJC_CLASS AVAsset;
OBJC_CLASS AVStreamDataParser;
OBJC_CLASS NSError;
OBJC_CLASS NSObject;
OBJC_PROTOCOL(WebSampleBufferVideoRendering);
typedef struct opaqueCMSampleBuffer *CMSampleBufferRef;

namespace WebCore {

class CDMInstance;
class LegacyCDMSession;
class MediaPlayerPrivateMediaSourceAVFObjC;
class MediaSourcePrivateClient;
class SourceBufferPrivateAVFObjC;
class TimeRanges;
class VideoMediaSampleRenderer;

class MediaSourcePrivateAVFObjC final
    : public MediaSourcePrivate
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
{
public:
    static Ref<MediaSourcePrivateAVFObjC> create(MediaPlayerPrivateMediaSourceAVFObjC&, MediaSourcePrivateClient&);
    virtual ~MediaSourcePrivateAVFObjC();

    constexpr MediaPlatformType platformType() const final { return MediaPlatformType::AVFObjC; }

    RefPtr<MediaPlayerPrivateInterface> player() const final;
    void setPlayer(MediaPlayerPrivateInterface*) final;

    AddStatus addSourceBuffer(const ContentType&, RefPtr<SourceBufferPrivate>&) final;
    void durationChanged(const MediaTime&) final;
    void markEndOfStream(EndOfStreamStatus) final;

    MediaPlayer::ReadyState mediaPlayerReadyState() const final;
    void setMediaPlayerReadyState(MediaPlayer::ReadyState) final;

    bool hasSelectedVideo() const;

    void willSeek();

    FloatSize naturalSize() const;

    void hasSelectedVideoChanged(SourceBufferPrivateAVFObjC&);
    void setVideoRenderer(VideoMediaSampleRenderer*);
    void stageVideoRenderer(VideoMediaSampleRenderer*);
    void videoRendererWillReconfigure(VideoMediaSampleRenderer&);
    void videoRendererDidReconfigure(VideoMediaSampleRenderer&);

    void flushActiveSourceBuffersIfNeeded();

#if ENABLE(ENCRYPTED_MEDIA)
    void cdmInstanceAttached(CDMInstance&);
    void cdmInstanceDetached(CDMInstance&);
    void attemptToDecryptWithInstance(CDMInstance&);
    bool waitingForKey() const;

    CDMInstance* cdmInstance() const { return m_cdmInstance.get(); }
    void outputObscuredDueToInsufficientExternalProtectionChanged(bool);
#endif

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger.get(); }
    ASCIILiteral logClassName() const final { return "MediaSourcePrivateAVFObjC"_s; }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    WTFLogChannel& logChannel() const final;

    uint64_t nextSourceBufferLogIdentifier() { return childLogIdentifier(m_logIdentifier, ++m_nextSourceBufferID); }
#endif

    using RendererType = MediaSourcePrivateClient::RendererType;
    void failedToCreateRenderer(RendererType);
    bool needsVideoLayer() const;

    void setResourceOwner(const ProcessIdentity& resourceOwner) { m_resourceOwner = resourceOwner; }

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)
    void keyAdded();
#endif

private:
    friend class SourceBufferPrivateAVFObjC;

    MediaSourcePrivateAVFObjC(MediaPlayerPrivateMediaSourceAVFObjC&, MediaSourcePrivateClient&);
    MediaPlayerPrivateMediaSourceAVFObjC* platformPlayer() const { return m_player.get(); }

    void notifyActiveSourceBuffersChanged() final;
#if ENABLE(LEGACY_ENCRYPTED_MEDIA)
    void sourceBufferKeyNeeded(SourceBufferPrivateAVFObjC*, const SharedBuffer&);
#endif
    void removeSourceBuffer(SourceBufferPrivate&) final;

    void setSourceBufferWithSelectedVideo(SourceBufferPrivateAVFObjC*);

    void bufferedChanged(const PlatformTimeRanges&) final;
    void trackBufferedChanged(SourceBufferPrivate&, Vector<PlatformTimeRanges>&&) final;

    WeakPtr<MediaPlayerPrivateMediaSourceAVFObjC> m_player;
    Deque<SourceBufferPrivateAVFObjC*> m_sourceBuffersNeedingSessions;
    SourceBufferPrivateAVFObjC* m_sourceBufferWithSelectedVideo { nullptr };
#if ENABLE(ENCRYPTED_MEDIA)
    RefPtr<CDMInstance> m_cdmInstance;
#endif
#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
    uint64_t m_nextSourceBufferID { 0 };
#endif

    UncheckedKeyHashMap<SourceBufferPrivate*, Vector<PlatformTimeRanges>> m_bufferedRanges;
    ProcessIdentity m_resourceOwner;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::MediaSourcePrivateAVFObjC)
static bool isType(const WebCore::MediaSourcePrivate& mediaSource) { return mediaSource.platformType() == WebCore::MediaPlatformType::AVFObjC; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(MEDIA_SOURCE) && USE(AVFOUNDATION)
