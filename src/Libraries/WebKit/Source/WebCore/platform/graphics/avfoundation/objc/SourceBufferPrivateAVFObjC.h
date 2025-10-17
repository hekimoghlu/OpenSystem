/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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
#include "SourceBufferParser.h"
#include "SourceBufferPrivate.h"
#include "WebAVSampleBufferListener.h"
#include <dispatch/group.h>
#include <wtf/Box.h>
#include <wtf/CancellableTask.h>
#include <wtf/Deque.h>
#include <wtf/LoggerHelper.h>
#include <wtf/MediaTime.h>
#include <wtf/OSObjectPtr.h>
#include <wtf/Observer.h>
#include <wtf/RefPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/Vector.h>
#include <wtf/WTFSemaphore.h>
#include <wtf/WeakPtr.h>
#include <wtf/threads/BinarySemaphore.h>

OBJC_CLASS AVStreamDataParser;
OBJC_CLASS AVSampleBufferAudioRenderer;
OBJC_CLASS NSData;
OBJC_CLASS NSError;
OBJC_CLASS NSObject;
OBJC_PROTOCOL(WebSampleBufferVideoRendering);

typedef struct opaqueCMSampleBuffer *CMSampleBufferRef;
typedef const struct opaqueCMFormatDescription *CMFormatDescriptionRef;

namespace WTF {
class WorkQueue;
}

namespace WebCore {

class CDMInstance;
class CDMInstanceFairPlayStreamingAVFObjC;
class CDMSessionAVContentKeySession;
class MediaPlayerPrivateMediaSourceAVFObjC;
class MediaSourcePrivateAVFObjC;
class TimeRanges;
class AudioTrackPrivate;
class VideoMediaSampleRenderer;
class VideoTrackPrivate;
class AudioTrackPrivateMediaSourceAVFObjC;
class VideoTrackPrivateMediaSourceAVFObjC;
class WebCoreDecompressionSession;
class SharedBuffer;

struct TrackInfo;

class SourceBufferPrivateAVFObjCErrorClient {
public:
    virtual ~SourceBufferPrivateAVFObjCErrorClient() = default;
    virtual void videoRendererDidReceiveError(WebSampleBufferVideoRendering *, NSError *, bool& shouldIgnore) = 0;
ALLOW_NEW_API_WITHOUT_GUARDS_BEGIN
    virtual void audioRendererDidReceiveError(AVSampleBufferAudioRenderer *, NSError *, bool& shouldIgnore) = 0;
ALLOW_NEW_API_WITHOUT_GUARDS_END
};

class SourceBufferPrivateAVFObjC final
    : public SourceBufferPrivate
    , public WebAVSampleBufferListenerClient
{
public:
    static Ref<SourceBufferPrivateAVFObjC> create(MediaSourcePrivateAVFObjC&, Ref<SourceBufferParser>&&);
    virtual ~SourceBufferPrivateAVFObjC();

    constexpr MediaPlatformType platformType() const final { return MediaPlatformType::AVFObjC; }

    void didProvideContentKeyRequestInitializationDataForTrackID(Ref<SharedBuffer>&&, TrackID, Box<BinarySemaphore>);

    void didProvideContentKeyRequestIdentifierForTrackID(Ref<SharedBuffer>&&, TrackID);

    bool hasSelectedVideo() const;

    void trackDidChangeSelected(VideoTrackPrivate&, bool selected);
    void trackDidChangeEnabled(AudioTrackPrivate&, bool enabled);

    void willSeek();
    void seekToTime(const MediaTime&) final;
    FloatSize naturalSize();

    const std::optional<TrackID>& protectedTrackID() const { return m_protectedTrackID; }
    bool needsVideoLayer() const;

#if (ENABLE(ENCRYPTED_MEDIA) && HAVE(AVCONTENTKEYSESSION)) || ENABLE(LEGACY_ENCRYPTED_MEDIA)
    AVStreamDataParser* streamDataParser() const { return m_streamDataParser.get(); }
    void setCDMSession(LegacyCDMSession*) final;
    void setCDMInstance(CDMInstance*) final;
    void attemptToDecrypt() final;
    bool waitingForKey() const final { return m_waitingForKey; }
#endif

    void flush();
    void flushIfNeeded();

    void registerForErrorNotifications(SourceBufferPrivateAVFObjCErrorClient*);
    void unregisterForErrorNotifications(SourceBufferPrivateAVFObjCErrorClient*);

    void setVideoRenderer(VideoMediaSampleRenderer*);
    void stageVideoRenderer(VideoMediaSampleRenderer*);
    void videoRendererWillReconfigure(VideoMediaSampleRenderer&);
    void videoRendererDidReconfigure(VideoMediaSampleRenderer&);

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)
    SharedBuffer* initData() { return m_initData.get(); }
#endif

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger.get(); }
    ASCIILiteral logClassName() const override { return "SourceBufferPrivateAVFObjC"_s; }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    WTFLogChannel& logChannel() const final;
    const Logger& sourceBufferLogger() const final { return m_logger.get(); }
    uint64_t sourceBufferLogIdentifier() final { return logIdentifier(); }
#endif

    void setResourceOwner(const ProcessIdentity& resourceOwner) { m_resourceOwner = resourceOwner; }

private:
    explicit SourceBufferPrivateAVFObjC(MediaSourcePrivateAVFObjC&, Ref<SourceBufferParser>&&);

    void didProvideMediaDataForTrackId(Ref<MediaSampleAVFObjC>&&, TrackID, const String& mediaType);
    bool isMediaSampleAllowed(const MediaSample&) const final;

    // SourceBufferPrivate overrides
    Ref<MediaPromise> appendInternal(Ref<SharedBuffer>&&) final;
    void abort() final;
    void resetParserStateInternal() final;
    void removedFromMediaSource() final;
    void flush(TrackID) final;
    void enqueueSample(Ref<MediaSample>&&, TrackID) final;
    bool isReadyForMoreSamples(TrackID) final;
    MediaTime timeFudgeFactor() const final;
    void notifyClientWhenReadyForMoreSamples(TrackID) final;
    bool canSetMinimumUpcomingPresentationTime(TrackID) const override;
    void setMinimumUpcomingPresentationTime(TrackID, const MediaTime&) override;
    void clearMinimumUpcomingPresentationTime(TrackID) override;
    bool canSwitchToType(const ContentType&) final;
    bool isSeeking() const final;

    bool precheckInitializationSegment(const InitializationSegment&) final;
    void processInitializationSegment(std::optional<InitializationSegment>&&) final;
    void processFormatDescriptionForTrackId(Ref<TrackInfo>&&, TrackID) final;
    void updateTrackIds(Vector<std::pair<TrackID, TrackID>>&&) final;

    // WebAVSampleBufferListenerClient
    void videoRendererDidReceiveError(WebSampleBufferVideoRendering *, NSError *) final;
    void audioRendererWasAutomaticallyFlushed(AVSampleBufferAudioRenderer *, const CMTime&) final;
    void outputObscuredDueToInsufficientExternalProtectionChanged(bool) final;
    void videoRendererRequiresFlushToResumeDecodingChanged(WebSampleBufferVideoRendering *, bool) final;
    void videoRendererReadyForDisplayChanged(WebSampleBufferVideoRendering *, bool isReadyForDisplay) final;
    void audioRendererDidReceiveError(AVSampleBufferAudioRenderer *, NSError *) final;

    void processPendingTrackChangeTasks();
    void enqueueSample(Ref<MediaSampleAVFObjC>&&, TrackID);
    void enqueueSampleBuffer(MediaSampleAVFObjC&);
    void attachContentKeyToSampleIfNeeded(const MediaSampleAVFObjC&);
    void didBecomeReadyForMoreSamples(TrackID);
    void appendCompleted(bool);
    void destroyStreamDataParser();
    void destroyRenderers();
    void clearTracks();

    bool isEnabledVideoTrackID(TrackID) const;
    bool requiresFlush() const;
    void flushVideo();
ALLOW_NEW_API_WITHOUT_GUARDS_BEGIN
    RetainPtr<AVSampleBufferAudioRenderer> audioRendererForTrackID(TrackID) const;
    void flushAudio(AVSampleBufferAudioRenderer *);
ALLOW_NEW_API_WITHOUT_GUARDS_END

    RefPtr<MediaPlayerPrivateMediaSourceAVFObjC> player() const;
    bool canEnqueueSample(TrackID, const MediaSampleAVFObjC&);
    bool trackIsBlocked(TrackID) const;

#if ENABLE(ENCRYPTED_MEDIA) && HAVE(AVCONTENTKEYSESSION)
    void tryToEnqueueBlockedSamples();
#endif

    void setTrackChangeCallbacks(const InitializationSegment&, bool initialized);

    void configureVideoRenderer(VideoMediaSampleRenderer&);
    void invalidateVideoRenderer(VideoMediaSampleRenderer&);

    StdUnorderedMap<TrackID, RefPtr<VideoTrackPrivate>> m_videoTracks;
    StdUnorderedMap<TrackID, RefPtr<AudioTrackPrivate>> m_audioTracks;
    Vector<SourceBufferPrivateAVFObjCErrorClient*> m_errorClients;

    const Ref<SourceBufferParser> m_parser;
    Vector<Function<void()>> m_pendingTrackChangeTasks;
    Deque<std::pair<TrackID, Ref<MediaSampleAVFObjC>>> m_blockedSamples;

    RefPtr<VideoMediaSampleRenderer> m_videoRenderer;
    RefPtr<VideoMediaSampleRenderer> m_expiringVideoRenderer;
ALLOW_NEW_API_WITHOUT_GUARDS_BEGIN
    StdUnorderedMap<TrackID, RetainPtr<AVSampleBufferAudioRenderer>> m_audioRenderers;
ALLOW_NEW_API_WITHOUT_GUARDS_END
    const Ref<WebAVSampleBufferListener> m_listener;
#if PLATFORM(IOS_FAMILY)
    bool m_displayLayerWasInterrupted { false };
#endif
    RetainPtr<NSError> m_hdcpError;
    Box<BinarySemaphore> m_hasSessionSemaphore;
    Box<Semaphore> m_abortSemaphore;
    const Ref<WTF::WorkQueue> m_appendQueue;

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)
    RefPtr<SharedBuffer> m_initData;
    WeakPtr<CDMSessionAVContentKeySession> m_session { nullptr };
#endif
#if ENABLE(ENCRYPTED_MEDIA) && HAVE(AVCONTENTKEYSESSION)
    using KeyIDs = Vector<Ref<SharedBuffer>>;
    struct TrackInitData {
        RefPtr<SharedBuffer> initData;
        KeyIDs keyIDs;
    };
    using TrackInitDataMap = StdUnorderedMap<TrackID, TrackInitData>;
    TrackInitDataMap m_pendingProtectedTrackInitDataMap;
    TrackInitDataMap m_protectedTrackInitDataMap;

    using TrackKeyIDsMap = StdUnorderedMap<TrackID, KeyIDs>;
    TrackKeyIDsMap m_currentTrackIDs;

    RefPtr<CDMInstanceFairPlayStreamingAVFObjC> m_cdmInstance;
    UniqueRef<Observer<void()>> m_keyStatusesChangedObserver;
    KeyIDs m_keyIDs;
#endif

    std::optional<FloatSize> m_cachedSize;
    FloatSize m_currentSize;
    bool m_waitingForKey { true };
    bool m_seeking { false };
    bool m_layerRequiresFlush { false };
    std::optional<TrackID> m_enabledVideoTrackID;
    std::optional<TrackID> m_protectedTrackID;
#if ENABLE(ENCRYPTED_MEDIA) && HAVE(AVCONTENTKEYSESSION)
    RetainPtr<AVStreamDataParser> m_streamDataParser;
#endif

#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif

    ProcessIdentity m_resourceOwner;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SourceBufferPrivateAVFObjC)
static bool isType(const WebCore::SourceBufferPrivate& sourceBuffer) { return sourceBuffer.platformType() == WebCore::MediaPlatformType::AVFObjC; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(MEDIA_SOURCE) && USE(AVFOUNDATION)
