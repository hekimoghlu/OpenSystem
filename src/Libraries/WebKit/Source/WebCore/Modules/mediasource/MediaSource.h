/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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

#if ENABLE(MEDIA_SOURCE)

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include "ExceptionOr.h"
#include "HTMLMediaElement.h"
#include "MediaPlayer.h"
#include "MediaPromiseTypes.h"
#include "MediaSourceInit.h"
#include "MediaSourcePrivateClient.h"
#include "URLRegistry.h"
#include <optional>
#include <wtf/LoggerHelper.h>
#include <wtf/NativePromise.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class AudioTrack;
class AudioTrackPrivate;
class ContentType;
class InbandTextTrackPrivate;
class MediaSourceClientImpl;
class MediaSourceHandle;
class SourceBuffer;
class SourceBufferList;
class SourceBufferPrivate;
class TextTrack;
class TimeRanges;
class VideoTrack;
class VideoTrackPrivate;

enum class MediaSourceReadyState { Closed, Open, Ended };

class MediaSource
    : public RefCounted<MediaSource>
    , public CanMakeWeakPtr<MediaSource>
    , public ActiveDOMObject
    , public EventTarget
    , public URLRegistrable
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
    , private Logger::Observer
#endif
{
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MediaSource);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    USING_CAN_MAKE_WEAKPTR(CanMakeWeakPtr<MediaSource>);

    static void setRegistry(URLRegistry*);
    static MediaSource* lookup(const String& url) { return s_registry ? downcast<MediaSource>(s_registry->lookup(url)) : nullptr; }

    static Ref<MediaSource> create(ScriptExecutionContext&, MediaSourceInit&&);
    virtual ~MediaSource();

    static bool enabledForContext(ScriptExecutionContext&);

    void addedToRegistry();
    void removedFromRegistry();
    void openIfInEndedState();
    void openIfDeferredOpen();
    bool isOpen() const;
    virtual void monitorSourceBuffers();
    bool isClosed() const;
    bool isEnded() const;
    void sourceBufferDidChangeActiveState(SourceBuffer&, bool);
    MediaTime duration() const;
    PlatformTimeRanges buffered() const;

    enum class EndOfStreamError { Network, Decode };
    void streamEndedWithError(std::optional<EndOfStreamError>);

    bool attachToElement(WeakPtr<HTMLMediaElement>&&);
    void elementIsShuttingDown();
    void detachFromElement();
    bool isSeeking() const { return !!m_pendingSeekTarget; }
    Ref<TimeRanges> seekable();
    ExceptionOr<void> setLiveSeekableRange(double start, double end);
    ExceptionOr<void> clearLiveSeekableRange();

    ExceptionOr<void> setDuration(double);
    ExceptionOr<void> setDurationInternal(const MediaTime&);
    MediaTime currentTime() const;

    using ReadyState = MediaSourceReadyState;
    ReadyState readyState() const;
    ExceptionOr<void> endOfStream(std::optional<EndOfStreamError>);

    Ref<SourceBufferList> sourceBuffers() const;
    Ref<SourceBufferList> activeSourceBuffers() const;
    ExceptionOr<Ref<SourceBuffer>> addSourceBuffer(const String& type);
    ExceptionOr<void> removeSourceBuffer(SourceBuffer&);
    static bool isTypeSupported(ScriptExecutionContext&, const String& type);

#if ENABLE(MEDIA_SOURCE_IN_WORKERS)
    Ref<MediaSourceHandle> handle();
    static bool canConstructInDedicatedWorker(ScriptExecutionContext&);
    void registerTransferredHandle(MediaSourceHandle&);
#endif
    bool detachable() const { return m_detachable; }

    ScriptExecutionContext* scriptExecutionContext() const final;

    static const MediaTime& currentTimeFudgeFactor();
    static bool contentTypeShouldGenerateTimestamps(const ContentType&);

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger.get(); }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    ASCIILiteral logClassName() const final { return "MediaSource"_s; }
    WTFLogChannel& logChannel() const final;
    void setLogIdentifier(uint64_t);

    Ref<Logger> logger(ScriptExecutionContext&);
    void didLogMessage(const WTFLogChannel&, WTFLogLevel, Vector<JSONLogValue>&&) final;
#endif

    virtual bool isManaged() const { return false; }
    virtual bool streaming() const { return false; }
    void memoryPressure();

    void setAsSrcObject(bool);

    // Called by SourceBuffer.
    void sourceBufferBufferedChanged();
    void sourceBufferReceivedFirstInitializationSegmentChanged();
    void sourceBufferActiveTrackFlagChanged(bool);
    void setMediaPlayerReadyState(MediaPlayer::ReadyState);
    void incrementDroppedFrameCount();
    void addAudioTrackToElement(Ref<AudioTrack>&&);
    void addTextTrackToElement(Ref<TextTrack>&&);
    void addVideoTrackToElement(Ref<VideoTrack>&&);
    void addAudioTrackMirrorToElement(Ref<AudioTrackPrivate>&&, bool enabled);
    void addTextTrackMirrorToElement(Ref<InbandTextTrackPrivate>&&);
    void addVideoTrackMirrorToElement(Ref<VideoTrackPrivate>&&, bool selected);

    Ref<MediaSourcePrivateClient> client() const;

protected:
    MediaSource(ScriptExecutionContext&, MediaSourceInit&&);

    bool isBuffered(const PlatformTimeRanges&) const;

    void scheduleEvent(const AtomString& eventName);
    void notifyElementUpdateMediaState() const;
    void ensureWeakOnHTMLMediaElementContext(Function<void(HTMLMediaElement&)>&&) const;

    virtual void elementDetached() { }

    RefPtr<MediaSourcePrivate> protectedPrivate() const;

    WeakPtr<HTMLMediaElement> m_mediaElement;
    bool m_detachable { false };

private:
    friend class MediaSourceClientImpl;

    // ActiveDOMObject.
    void stop() final;
    bool virtualHasPendingActivity() const final;

    static bool isTypeSupported(ScriptExecutionContext&, const String& type, Vector<ContentType>&& contentTypesRequiringHardwareSupport);

    void setPrivate(RefPtr<MediaSourcePrivate>&&);
    void setPrivateAndOpen(Ref<MediaSourcePrivate>&&);
    void reOpen();
    void open();

    void removeSourceBufferWithOptionalDestruction(SourceBuffer&, bool withDestruction);

    Ref<MediaTimePromise> waitForTarget(const SeekTarget&);
    Ref<MediaPromise> seekToTime(const MediaTime&);
    using RendererType = MediaSourcePrivateClient::RendererType;
    void failedToCreateRenderer(RendererType);

    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
    enum EventTargetInterfaceType eventTargetInterface() const final;

    // URLRegistrable.
    URLRegistry& registry() const final;
    RegistrableType registrableType() const final { return RegistrableType::MediaSource; }

    void setReadyState(ReadyState);
    void onReadyStateChange(ReadyState oldState, ReadyState newState);

    Vector<PlatformTimeRanges> activeRanges() const;

    ExceptionOr<Ref<SourceBufferPrivate>> createSourceBufferPrivate(const ContentType&);

    void regenerateActiveSourceBuffers();
    void updateBufferedIfNeeded(bool forced = false);

    bool hasBufferedTime(const MediaTime&);
    bool hasCurrentTime();
    bool hasFutureTime();

    void completeSeek();

    static URLRegistry* s_registry;

    const Ref<SourceBufferList> m_sourceBuffers;
    const Ref<SourceBufferList> m_activeSourceBuffers;
    std::optional<SeekTarget> m_pendingSeekTarget;
    std::optional<MediaTimePromise::AutoRejectProducer> m_seekTargetPromise;
    bool m_openDeferred { false };
    bool m_sourceopenPending { false };
    bool m_isAttached { false };
    std::optional<ReadyState> m_readyStateBeforeDetached;
#if ENABLE(MEDIA_SOURCE_IN_WORKERS)
    RefPtr<MediaSourceHandle> m_handle;
#endif

#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    uint64_t m_logIdentifier { 0 };
#endif
    std::atomic<uint64_t> m_associatedRegistryCount { 0 };
    RefPtr<MediaSourcePrivate> m_private;
    Ref<MediaSourceClientImpl> m_client;
};

String convertEnumerationToString(MediaSource::EndOfStreamError);
String convertEnumerationToString(MediaSource::ReadyState);

} // namespace WebCore

namespace WTF {

template<typename Type>
struct LogArgument;

template <>
struct LogArgument<WebCore::MediaSource::EndOfStreamError> {
    static String toString(const WebCore::MediaSource::EndOfStreamError error)
    {
        return convertEnumerationToString(error);
    }
};

template <>
struct LogArgument<WebCore::MediaSource::ReadyState> {
    static String toString(const WebCore::MediaSource::ReadyState state)
    {
        return convertEnumerationToString(state);
    }
};

} // namespace WTF

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::MediaSource)
    static bool isType(const WebCore::URLRegistrable& registrable) { return registrable.registrableType() == WebCore::URLRegistrable::RegistrableType::MediaSource; }
SPECIALIZE_TYPE_TRAITS_END()

#endif
